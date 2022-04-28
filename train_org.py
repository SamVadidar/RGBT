import argparse
import math
import os
import random
import time
from pathlib import Path
from warnings import warn

import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test_org
from Fusion.models.models import *
from Fusion.utils.datasets import create_dataloader
from Fusion.utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, fitness_p, fitness_r, fitness_ap50, fitness_ap, fitness_f, strip_optimizer,\
    print_mutation, set_logging, check_img_size
from Fusion.utils.loss import compute_loss
from Fusion.utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from Fusion.utils.torch_utils import ModelEMA, select_device
from FLIR_PP.arg_parser import DATASET_PP_PATH

logger = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    wandb = None
    logger.info("Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")

def weight_sanity(ckpt, dict):
    if ckpt['epoch'] == -1:
        weights, _ = os.path.split(dict['weight_path'])
        for file in Path(weights).rglob('*.pt'):
            if os.path.getsize(str(file)) > 400000000: # 400MB
                print('Weight Path is replaced with: ', str(file))
                ckpt = torch.load(str(file), map_location='cuda')
                print('Saved @ epoch: ', ckpt['epoch'])
                return ckpt, str(file)
    else:
        print('Saved @ epoch: ', ckpt['epoch'])
        return ckpt, dict['weight_path']

def train(dict_, hyp, tb_writer=None, wandb=None, budget = None):
    logger = logging.getLogger(__name__)
    if dict_['evolve']:
        logger = logging.getLogger(__name__)
        dict_['epoch'] = budget

    logger.info(f'Hyperparameters {hyp}')
    log_dir = Path(tb_writer.log_dir) if tb_writer else Path(dict_['log_dir']) / 'evolve'  # logging directory
    save_dir, epochs, batch_size, total_batch_size, weights, rank = \
        Path(dict_['project']), dict_['epochs'], dict_['batch_size'], dict_['batch_size'], dict_['weight_path'], dict_['global_rank']

    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # Configure
    plots = not dict_['evolve']  # create plots
    device = dict_['device']
    cuda = device != 'cpu'
    init_seeds(2 + rank)

    train_path = dict_['train_path']
    test_path = dict_['test_path'] if dict_['task']=='test' else dict_['val_path']
    nc = dict_['nclasses']
    names = dict_['names']
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, train_path)  # check
    img_size = dict_['img_size']

    # Model
    pretrained = weights.endswith('.pt')
    if pretrained:
        try:
            ckpt = torch.load(weights, map_location=device)  # load checkpoint
        except:
            weights_path, _ = os.path.split(dict_['weight_path'])
            for file in Path(weights_path).rglob('*.pt'):
                if os.path.getsize(str(file)) > 400000000: # 400MB
                    ckpt = torch.load(str(file), map_location=device)  # load checkpoint

        # Making sure the weight file is not corrupted
        # if ckpt['epoch'] != None:
        #     ckpt, dict_['weight_path'] = weight_sanity(ckpt, dict_)
        if dict_['mode'] == 'fusion':
            model = Fused_Darknets(dict_, (img_size, img_size)).to(device) # create
        else:
            # Darknet input_size: ir = 1 and rgb = 3
            model = Darknet(dict_, (img_size, img_size)).to(device) # create
        state_dict = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(state_dict, strict=False)
    else:
        model = Darknet(dict_, (img_size, img_size)).to(device) # create

    with open(save_dir / 'dict.yaml', 'w') as f:
        yaml.dump(dict_, f, sort_keys=False)
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if 'backbone' in k and not 'f_x' in k and dict_['mode']=='fusion' and dict_['backbone_freeze']:
            v.requires_grad = False # freeze backbones
        if '.bias' in k:
            pg2.append(v)  # biases
        elif 'conv.weight' in k: # or '1.weight'in k:
            pg1.append(v)  # apply weight_decay
        elif k in ['head.final3.weight','head.final4.weight','head.final5.weight']:
            pg1.append(v)
        else:
            pg0.append(v)  # all else

    # if not dict_['warmup']:
    #     hyp['lr0'] = ckpt['optimizer']['param_groups'][0]['lr'] # replace initial lr with last lr of loaded weight

    if dict_['use_adam']:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp['lrf']) + hyp['lrf']  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.9)

    # Logging
    if wandb and wandb.run is None:
        # dict_.hyp = hyp  # add hyperparameters
        wandb_run = wandb.init(config=dict_, resume="allow",
                               project='BL',# if dict_['project'] == 'runs/train' else Path(dict_['project']).stem,
                               name=save_dir,#.stem,
                               id=ckpt.get('wandb_id') if 'ckpt' in locals() else None)

    # Resume
    start_epoch, best_fitness = 0, 0.0
    best_fitness_p, best_fitness_r, best_fitness_ap50, best_fitness_ap, best_fitness_f = 0.0, 0.0, 0.0, 0.0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None and dict_['resume']:
            optimizer.load_state_dict(ckpt['optimizer'])
            if not dict_['warmup']:
                for element in optimizer.param_groups:
                    element['initial_lr'] = element['lr']
            best_fitness = ckpt['best_fitness']
            best_fitness_p = ckpt['best_fitness_p']
            best_fitness_r = ckpt['best_fitness_r']
            best_fitness_ap50 = ckpt['best_fitness_ap50']
            best_fitness_ap = ckpt['best_fitness_ap']
            best_fitness_f = ckpt['best_fitness_f']
            epochs = dict_['epochs'] + ckpt['epoch'] # to continue to the correct epoch number
        elif not dict_['resume']:
            ckpt['epoch'] = 0
            ckpt['training_results']=None
            ckpt['optimizer']=None
            best_fitness = 0
            best_fitness_p = 0
            best_fitness_r = 0
            best_fitness_ap50 = 0
            best_fitness_ap = 0
            best_fitness_f = 0

        # Results
        if ckpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(ckpt['training_results'])  # write results.txt

        # Epochs
        start_epoch = ckpt['epoch'] if ckpt['epoch']!= None else 0
        # if epochs < start_epoch:
        #     print('\n%s has been trained for %g epochs. Fine-tuning for %g additional epochs.\n' %
        #           (weights, ckpt['epoch'], epochs))
        #     epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, state_dict

    # gs = 32 # grid size (max stride)
    gs = 64 #int(max(model.stride))  # grid size (max stride)

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model,device_ids=device) # no device_ids

    # SyncBatchNorm
    if dict_['sync_bn'] and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        print('Using SyncBatchNorm()')

    # Exponential moving average
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # DDP mode
    if cuda and rank != -1:
        model = DDP(model, device_ids=[dict_['local_rank']], output_device=(dict_['local_rank']))

    # Trainloader
    dataloader, dataset = create_dataloader(train_path, dict_['img_size'], batch_size, gs, hyp=hyp, augment=dict_['train_aug'],
                                            cache=dict_['cache_images'], rect=dict_['rect'], rank=rank,
                                            world_size=dict_['world_size'], workers=dict_['workers'], img_format=dict_['img_format'], mode=dict_['mode'])

    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, dict_['names'], nc - 1)

    # Testloader
    # Process 0
    if rank in [-1, 0]:
        ema.updates = start_epoch * nb // accumulate  # set EMA updates ***
        # local_rank is set to -1. Because only the first process is expected to do evaluation.
        testloader = create_dataloader(test_path, dict_['img_size'], batch_size, gs, hyp=hyp, augment= False,
                                       cache=dict_['cache_images'], rect=dict_['rect_val'], rank=-1, world_size=dict_['world_size'], workers=dict_['workers'], img_format=dict_['img_format'], mode=dict_['mode'])[0]

        if not dict_['resume']:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                plot_labels(labels, save_dir=save_dir)
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)
                if wandb:
                    wandb.log({"Labels": [wandb.Image(str(x), caption=x.name) for x in save_dir.glob('*labels*.png')]})

    # Model parameters
    model.nc = dict_['nclasses']  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    logger.info('Image sizes %g train, %g test\n'
                'Using %g dataloader workers\nLogging results to %s\n'
                'Starting training for %g epochs...' % (dict_['img_size'], dict_['img_size'], dataloader.num_workers, save_dir, epochs))

    # torch.save(model, wdir / 'init.pt')

    # epochs += 1 # because we started from our 3c.pt and there we set initial epoch to 1
    best_val_loss = 1
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)
        if dict_['image_weights']:
            # Generate indices
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # mean losses
        # mloss_tot = torch.zeros(1, device=device)  # total mean losses

        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'targets', 'img_size'))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
            if len(imgs.shape) == 3: #  ir mode
                imgs = torch.unsqueeze(imgs, axis=1)

            # Warmup
            # if ni <= nw and dict_['warmup']: # integrated batches <= warmup iterations: max(3 epochs, 1k iterations)
            if epoch <= start_epoch+hyp['warmup_epochs'] and dict_['warmup']: # integrated batches <= warmup iterations: max(3 epochs, 1k iterations)
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    try:
                        x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    except:
                        x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, hyp['lr0'] * lf(epoch)])

                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if dict_['multi_scale']:
                sz = random.randrange(dict_['img_size'] * 0.5, dict_['img_size'] * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with amp.autocast(enabled=cuda):
                if dict_['mode'] == 'fusion':
                    # ir_gui = imgs[0, :1, :, :]
                    # ir_gui = ir_gui.permute(1,2,0)
                    # ir_gui = torch.squeeze(ir_gui)
                    # ir_gui *= 255
                    # ir_gui = ir_gui.to(device='cpu').numpy()
                    # ir_gui = cv2.cvtColor(ir_gui, cv2.COLOR_BGR2RGB)
                    # cv2.imwrite('./ir_gui.jpg', ir_gui)

                    # rgb_gui = imgs[0, 1:, :, :]
                    # rgb_gui = rgb_gui.permute(1,2,0)
                    # rgb_gui = torch.squeeze(rgb_gui)
                    # rgb_gui *= 255
                    # rgb_gui = rgb_gui.to(device='cpu').numpy()
                    # rgb_gui = cv2.cvtColor(rgb_gui, cv2.COLOR_BGR2RGB)
                    # cv2.imwrite('./rgb_gui.jpg', rgb_gui)

                    pred = model(imgs[:, 1:, :, :], imgs[:, :1, :, :]) # model(BGR, IR)
                else:
                    pred = model(imgs)

                loss, loss_items = compute_loss(pred, targets.to(device), hyp, dict_)  # scaled by batch_size
                if rank != -1:
                    loss *= dict_['world_size']  # gradient averaged between devices in DDP mode

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer) # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                # mloss_tot = (mloss_tot * i + loss) / (i + 1)  # update mean total loss
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                # Plot
                # if plots and ni < 3:
                if plots and i < 30 and epoch==start_epoch:
                    # f = save_dir / f'train_batch{ni}.jpg'  # filename
                    f = str(save_dir) + f'/train_batch{ni}' + dict_['img_format']  # filename
                    result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                    if tb_writer and result is not None:
                        tb_writer.add_image(str(f), result, dataformats='HWC', global_step=epoch)
                elif plots and ni == 3 and wandb:
                    wandb.log({"Mosaics": [wandb.Image(str(x), caption=x.name) for x in save_dir.glob('train*.jpg')]})

            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step() # cos or lambda function
        # # for ReduceOnPlataue
        # try:
        #     scheduler.step(val_loss)
        # except:
        #     val_loss = 1
        #     scheduler.step(val_loss)


        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            if ema:
                ema.update_attr(model)
            final_epoch = epoch + 1 == epochs
            if dict_['test_all'] or final_epoch:  # Calculate mAP
                results, maps, times = test_org.test(dict_,
                                                hyp,
                                                model = ema.ema.module if hasattr(ema.ema, 'module') else ema.ema,
                                                augment=False,
                                                dataloader=testloader,
                                                save_dir = save_dir,
                                                log_imgs=dict_['log_imgs'] if wandb else 0)

            # Write
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)

            # Tensorboard
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                    'x/lr0', 'x/lr1', 'x/lr2']  # params
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                if wandb:
                    wandb.log({tag: x})  # W&B
            tb_writer.add_scalar('train/loss', mloss[3], epoch)
            val_loss = results[4]+results[5]+results[6]
            tb_writer.add_scalar('val/loss', val_loss, epoch)

            # if tb_writer:
                # tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                #         'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                #         'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
                # for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                #     tb_writer.add_scalar(tag, x, epoch)

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            fi_p = fitness_p(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            fi_r = fitness_r(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            fi_ap50 = fitness_ap50(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            fi_ap = fitness_ap(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if (fi_p > 0.0) or (fi_r > 0.0):
                fi_f = fitness_f(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            else:
                fi_f = 0.0
            if fi > best_fitness:
                best_fitness = fi
            if fi_p > best_fitness_p:
                best_fitness_p = fi_p
            if fi_r > best_fitness_r:
                best_fitness_r = fi_r
            if fi_ap50 > best_fitness_ap50:
                best_fitness_ap50 = fi_ap50
            if fi_ap > best_fitness_ap:
                best_fitness_ap = fi_ap
            if fi_f > best_fitness_f:
                best_fitness_f = fi_f

            # Save model
            save = (dict_['save_all']!=False) or (final_epoch and not dict_['evolve'])
            if save:
                with open(results_file, 'r') as f:  # create checkpoint
                    ckpt = {'epoch': epoch,
                            'best_fitness': best_fitness,
                            'best_fitness_p': best_fitness_p,
                            'best_fitness_r': best_fitness_r,
                            'best_fitness_ap50': best_fitness_ap50,
                            'best_fitness_ap': best_fitness_ap,
                            'best_fitness_f': best_fitness_f,
                            'training_results': f.read(),
                            'model': ema.ema.module.state_dict() if hasattr(ema, 'module') else ema.ema.state_dict(),
                            'optimizer': None if final_epoch else optimizer.state_dict(),
                            'wandb_id': wandb_run.id if wandb else None}

                # Save last, best and delete
                # torch.save(ckpt, last)
                # if best_fitness == fi:
                #     torch.save(ckpt, best)
                # if (best_fitness == fi) and (epoch >= 200):
                #     torch.save(ckpt, wdir / 'best_{:03d}.pt'.format(epoch))
                if val_loss<best_val_loss:
                    best_val_loss = val_loss
                    torch.save(ckpt, wdir / 'best_val_loss.pt')
                # if best_fitness == fi:
                #     torch.save(ckpt, wdir / 'best_overall.pt')
                if best_fitness_p == fi_p:
                    torch.save(ckpt, wdir / 'best_p.pt')
                if best_fitness_r == fi_r:
                    torch.save(ckpt, wdir / 'best_r.pt')
                if best_fitness_ap50 == fi_ap50:
                    torch.save(ckpt, wdir / 'best_ap50.pt')
                # if best_fitness_ap == fi_ap:
                #     torch.save(ckpt, wdir / 'best_ap.pt')
                # if best_fitness_f == fi_f:
                #     torch.save(ckpt, wdir / 'best_f.pt')
                # if epoch == 0:
                #     torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                # if ((epoch+1) % 25) == 0:
                #     torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                if epoch >= (epochs-4):
                    torch.save(ckpt, wdir / 'last_{:03d}.pt'.format(epoch))
                # if epoch == (epochs+start_epoch):
                #     torch.save(ckpt, wdir / 'last.pt')
                # elif epoch >= 420:
                #     torch.save(ckpt, wdir / 'last_{:03d}.pt'.format(epoch))
                del ckpt

                # # Save last, best and delete
                # torch.save(ckpt, last)
                # if epoch >= (epochs-5):
                #     torch.save(ckpt, last.replace('.pt','_{:03d}.pt'.format(epoch)))
                # if (best_fitness == fi) and not final_epoch:
                #     # Delete previous best
                #     for weight in Path(wdir).rglob('*.pt'):
                #         _, weight_name = os.path.split(weight)
                #         if str(weight_name).startswith('best'): os.remove(str(weight))
                #     # Save the current best
                #     torch.save(ckpt, best.replace('.pt','_{:03d}.pt'.format(best_epoch)))

                # del ckpt
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    if rank in [-1, 0]:
        # Strip optimizers
        n = dict_['names'] if dict_['names'][0].isnumeric() else ''
        fresults, flast, fbest = save_dir / f'results{n}.txt', wdir / f'last{n}.pt', wdir / f'best{n}.pt'
        for f1, f2 in zip([wdir / 'last.pt', wdir / 'best.pt', results_file], [flast, fbest, fresults]):
            if f1.exists():
                os.rename(f1, f2)  # rename
                if str(f2).endswith('.pt'):  # is *.pt
                    strip_optimizer(f2)  # strip optimizer
        # Finish
        if plots:
            plot_results(save_dir=save_dir)  # save as results.png
            if wandb:
                wandb.log({"Results": [wandb.Image(str(save_dir / x), caption=x) for x in
                                       ['results.png', 'precision-recall_curve.png']]})
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    else:
        dist.destroy_process_group()
    #     # Finish
    #     if not dict_['evolve']:
    #         plot_results(save_dir=log_dir)  # save as results.png
    #     print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))

    # dist.destroy_process_group() if rank not in [-1, 0] else None

    wandb.run.finish() if wandb and wandb.run else None
    torch.cuda.empty_cache()

    if dict_['evolve']:
        return {"fitness":1/fi_ap50, "cost": time.time()-t0, "info":{"AP50":results[2], "budget":dict_['epochs']}}
    else:
        return results


if __name__ == '__main__':
    dict_ = {
        'device':'cuda', #Intialise device as cpu. Later check if cuda is avaialbel and change to cuda
        'device_num': '0',

        # Kmeans on COCO
        'anchors_g': [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]],
        'nclasses': 3, #Number of classes
        'names' : ['person', 'bicycle', 'car'],
        # 'gs': 32, #Image size multiples
        'img_size': 640, #Input image size. Must be a multiple of 32
        'strides': [8,16,32], #strides of p3,p4,p5
        'epochs': 150, #number of epochs
        'batch_size': 16, #train batch size
        'test_size': 16, #test batch size
        'use_adam': False, #Bool to use Adam optimiser
        'use_ema': True, #Exponential moving average control
        'multi_scale': True, #Bool to do multi-scale training
        'gr' : 1.0, # giou loss ratio (obj_loss = 1.0 or giou)
        'nms_conf_t':0.001, #0.2 Confidence training threshold
        'nms_merge': True, # it is passed to the test function
        'half': False,  # half precision for test/val set (only supported on CUDA)

        #logs
        'project': './runs/train',
        # 'logdir': './miniRuns',
        'comment': '_Series',
        'test_all': True, #Run test after end of each epoch
        'save_all': True, #Save checkpoints after every epoch
        'plot': True,
        'log_imgs': 16,
        'resume': False, # put epoch, training_results and optimizer to initial state
        # 'resume_lr': False, # defines the lr starting point
        'warmup': False, # controls the warmup, if False -> resume lr

        # DP
        'global_rank': -1, # -1
        'world_size': 1,
        'local_rank': -1, # -1
        'sync_bn': False,

        # Data loader
        'workers': 8,
        'cache_images': True,
        'rect': False, # train_set
        'rect_val': True, # val_set
        'image_weights': True,
        'img_format': '.jpg',
        'train_aug' : True,
        'mode': 'fusion', # 3 modes available: ir , rgb, fusion
        'backbone_freeze': False,

        # Hyp. Para.
        'evolve': False,

        # Modules
        'H_attention_bc' : False, # entropy based att. before concat.
        'H_attention_ac' : False, # entropy based att. after concat.
        'spatial': False, # spatial attention off/on (channel is always by default on!)

        # PATH
        'weight_path': './yolo_pre_3c.pt',
        'task': 'val',
        'train_path': DATASET_PP_PATH + '/Train_Test_Split/train/',
        'val_path': DATASET_PP_PATH + '/Train_Test_Split/dev/',

        'cam': False,
     }

    hyp = {
        'lr0': 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
        'lrf': 0.2, #final OneCycleLR learning rate (lr0 * lrf)
        'momentum': 0.937,  # SGD momentum/Adam beta1
        'weight_decay': 0.0005,  # optimizer weight decay
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8, # warmup initial momentum
        'warmup_bias_lr': 0.1, # warmup initial bias lr
        # 'giou': 0.05,  # GIoU loss gain
        'box': 0.05, # box loss gain
        'cls': 0.01875,  # cls loss gain | cls_org = 0.5 | ['cls'] *= nc / 80
        'cls_pw': 1.0,  # cls BCELoss positive_weight
        'obj': 1.0, # 0.7 # obj loss gain (scale with pixels)
        'obj_pw': 1.0,  # obj BCELoss positive_weight
        'iou_t': 0.6, # 0.2  # IoU training threshold
        'anchor_t': 4.0,  # anchor-multiple threshold
        'fl_gamma': 0.0,  # focal loss gamma (efficientDet default gamma=1.5)
        # To be Noted:
        # 1) degrees, translate, scale, shear, perspective work only if mosaic is off
        # 2) rect works only when image_weight is off
        # 3) mixup works only if aug is true and rect is false
        'hsv_h': 0.015,  # image HSV-Hue augmentation (fraction)
        'hsv_s': 0.7,  # image HSV-Saturation augmentation (fraction)
        'hsv_v': 0.4,  # image HSV-Value augmentation (fraction)
        'degrees': 0.0,  # image rotation (+/- deg)
        'translate': 0.0, # 0.1 # image translation (+/- fraction)
        'scale': 0.5,  # 0.9  #image scale (+/- gain)
        'shear': 0.0,  # image shear (+/- deg)
        'perspective': 0.0,  # image perspective (+/- fraction), range 0-0.001
        'flipud': 0.0,  # image flip up-down (probability)
        'fliplr': 0.5,  # image flip left-right (probability)
        'mosaic': 1.0,
        'mixup': 0.0, #mix up probability
    }


    # Condition to start the training on the server
    allowed_procs = 2
    sleep_period = 5 # in minutes
    cond = True

    # torch.cuda.list_gpu_processes(1)
    gp = torch.cuda.list_gpu_processes(int(dict_['device_num'])).split('\n')
    alc_procs = len(gp) - 1 # to remove gpu_index

    while(cond):
        # 0.5 since PID is counted with the processes
        if (alc_procs<=allowed_procs):
            # Set DDP variables
            dict_['world_size'] = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
            dict_['global_rank'] = int(os.environ['RANK']) if 'RANK' in os.environ else -1
            set_logging(dict_['global_rank'])
            # if dict_['global_rank'] in [-1, 0]:
            #     check_git_status()

            # # Resume
            # if dict_['resume']:
            #     last = get_latest_run() if dict_['resume'] == 'get_last' else dict_['resume']  # resume from most recent run
            #     if last and not dict_['weight_path']:
            #         print(f'Resuming training from {last}')
            #     dict_['weight_path'] = last if dict_['resume'] and not dict_['weight_path'] else dict_['weight_path']
            # else:
            #     dict_['project'] = increment_path(Path(dict_['project']) / ('exp'+dict_['comment']), exist_ok=False | dict_['evolve'])  # increment run

            device = select_device(dict_['device_num'], batch_size=dict_['batch_size'])
            total_batch_size = dict_['batch_size']

            # DDP mode
            if dict_['local_rank'] != -1:
                assert torch.cuda.device_count() > dict_['local_rank']
                # torch.cuda.set_device(dict_['local_rank'])
                torch.cuda.set_device(int(dict_['device_num']))
                # device = torch.device('cuda', dict_['local_rank'])
                device = torch.device('cuda', int(dict_['device_num']))
                dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
                dict_['world_size'] = dist.get_world_size()
                dict_['global_rank'] = dist.get_rank()
                assert dict_['batch_size'] % dict_['world_size'] == 0, '--batch-size must be multiple of CUDA device count'
                dict_['batch_size'] = dict_['total_batch_size'] // dict_['world_size']

            # Train
            if not dict_['evolve']:
                # tb_writer = None
                # if dict_['global_rank'] in [-1, 0]:
                #     tb_writer = SummaryWriter(dict_['project'])  # Tensorboard

                dict_['train_path'] = DATASET_PP_PATH + '/Train_Test_Split/train/'
                dict_['val_path'] = DATASET_PP_PATH + '/Train_Test_Split/dev/'

                # dict_['train_path'] = '/home/efs-gx/RGBT/CFR/val/'
                # dict_['val_path'] = '/home/efs-gx/RGBT/CFR/val'


#============================================================================================ RGB

                # dict_['img_size'] = 640
                # dict_['epochs'] = 150
                # dict_['batch_size'] = 8
                # dict_['test_size'] = 8
                # dict_['multi_scale'] = True
                # dict_['resume'] = False # for optimizer and epoch num.
                # dict_['warmup'] = True
                # dict_['mode'] = 'rgb'
                # dict_['img_format'] = '.jpg' if dict_['mode'] != 'ir' else '.jpeg'
                # hyp['mosaic'] = 1.0
                # hyp['mixup'] = 0.0
                # dict_['comment'] = '_RGB320_1000_RGB640_150'
                # dict_['weight_path'] = './runs/train/exp_RGB320_1000/weights/best_val_loss.pt'
                # dict_['project'] = './runs/train'
                # dict_['project'] = increment_path(Path(dict_['project']) / ('exp'+dict_['comment']), exist_ok=False | dict_['evolve'])  # increment run
                # tb_writer = None
                # tb_writer = SummaryWriter(dict_['project'])
                # train(dict_, hyp, tb_writer, wandb=False)


#============================================================================================ IR

                # dict_['img_size'] = 320
                # dict_['epochs'] = 300
                # dict_['batch_size'] = 32
                # dict_['test_size'] = 32
                # dict_['resume'] = False # for optimizer and epoch num.
                # dict_['warmup'] = True
                # dict_['mode'] = 'ir'
                # dict_['img_format'] = '.jpg' if dict_['mode'] != 'ir' else '.jpeg'
                # dict_['multi_scale'] = False
                # hyp['mosaic'] = 0.0
                # hyp['mixup'] = 0.0
                # dict_['attention_bc'] = False
                # dict_['H_attention_bc'] = False
                # dict_['comment'] = '_IR320_300noMSnoMos'
                # dict_['weight_path'] = '/home/ub145/Desktop/RGBT/IR.pt'
                # dict_['project'] = './runs/train'
                # dict_['project'] = increment_path(Path(dict_['project']) / ('exp'+dict_['comment']), exist_ok=False | dict_['evolve'])  # increment run
                # tb_writer = None
                # tb_writer = SummaryWriter(dict_['project'])
                # train(dict_, hyp, tb_writer, wandb=False)

#============================================================================================ Fusion

                # dict_['img_size'] = 320
                # dict_['epochs'] = 150
                # dict_['batch_size'] = 16
                # dict_['test_size'] = 16
                # dict_['warmup'] = True
                # dict_['resume'] = False # for optimizer and epoch num.
                # dict_['mode'] = 'fusion'
                # dict_['img_format'] = '.jpg' if dict_['mode'] != 'ir' else '.jpeg'
                # dict_['multi_scale'] = True
                # hyp['mosaic'] = 1.0
                # hyp['mixup'] = 0.0
                # dict_['H_attention_bc'] = True
                # dict_['H_attention_ac'] = True
                # dict_['spatial'] = True
                # dict_['comment'] = '_RGBT320_AlignedData_pre'
                # dict_['weight_path'] = './runs/train/exp_RGBT320_150_HACBC/weights/best_val_loss.pt'
                # dict_['backbone_freeze'] = False
                # dict_['project'] = './runs/train'
                # dict_['project'] = increment_path(Path(dict_['project']) / ('exp'+dict_['comment']), exist_ok=False | dict_['evolve'])  # increment run
                # tb_writer = None
                # tb_writer = SummaryWriter(dict_['project'])
                # train(dict_, hyp, tb_writer, wandb=False)

                dict_['img_size'] = 320
                dict_['epochs'] = 150
                dict_['batch_size'] = 8
                dict_['test_size'] = 8
                dict_['warmup'] = True
                dict_['resume'] = False # for optimizer and epoch num.
                dict_['mode'] = 'fusion'
                dict_['img_format'] = '.jpg' if dict_['mode'] != 'ir' else '.jpeg'
                dict_['multi_scale'] = True
                hyp['mosaic'] = 1.0
                hyp['mixup'] = 0.0
                dict_['H_attention_bc'] = True
                dict_['H_attention_ac'] = True
                dict_['spatial'] = True
                dict_['comment'] = '_RGBT320_EBAM'
                dict_['weight_path'] = './RGBT_pre.pt'
                dict_['backbone_freeze'] = False
                dict_['project'] = './runs/train'
                dict_['project'] = increment_path(Path(dict_['project']) / ('exp'+dict_['comment']), exist_ok=False | dict_['evolve'])  # increment run
                tb_writer = None
                tb_writer = SummaryWriter(dict_['project'])
                train(dict_, hyp, tb_writer, wandb=False)

        else:
            gp = torch.cuda.list_gpu_processes(int(dict_['device_num'])).split('\n')
            alc_procs = len(gp) - 1 # to remove gpu_index

            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            if alc_procs!=allowed_procs:
                print(alc_procs, 'Processes are running --- Number of allowed processes =', allowed_procs)
                print('GPU BUSY @ {}. Will try again in {} minutes'.format(current_time, sleep_period))
                time.sleep(sleep_period*60)
            else: print("Training is starting @ ", current_time)