import argparse
import math
import os
import random
import time
from pathlib import Path
from unicodedata import name

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

from test_org import test  # import test.py to get mAP after each epoch
from Fusion.models.models import *
from Fusion.utils.datasets import create_dataloader
from Fusion.utils.general import (
    check_img_size, torch_distributed_zero_first, labels_to_class_weights, plot_labels, check_anchors,
    labels_to_image_weights, compute_loss, plot_images, fitness, strip_optimizer, plot_results,
    get_latest_run, check_git_status, check_file, increment_dir, print_mutation, plot_evolution)
from Fusion.utils.google_utils import attempt_download
from Fusion.utils.torch_utils import init_seeds, ModelEMA, select_device, intersect_dicts
from FLIR_PP.arg_parser import DATASET_PP_PATH

def train(dict_, hyp, tb_writer=None):
    print(f'Hyperparameters {hyp}')
    log_dir = Path(tb_writer.log_dir) if tb_writer else Path(dict_['log_dir']) / 'evolve'  # logging directory
    wdir = str(log_dir / 'weights') + os.sep  # weights directory
    os.makedirs(wdir, exist_ok=True)
    last = wdir + 'last.pt'
    best = wdir + 'best.pt'
    results_file = str(log_dir / 'results.txt')
    epochs, batch_size, total_batch_size, weights, rank = \
        dict_['epochs'], dict_['batch_size'], dict_['batch_size'], dict_['weight_path'], dict_['global_rank']

    # # org below
    # TODO: Use DDP logging. Only the first process is allowed to log.
    # Save run settings
    # with open(log_dir / 'hyp.yaml', 'w') as f:
    #     yaml.dump(dict, f, sort_keys=False)
    # # org below
    # with open(log_dir / 'opt.yaml', 'w') as f:
    #     yaml.dump(vars(opt), f, sort_keys=False)

    with open(log_dir / 'dict.yaml', 'w') as f:
        yaml.dump(dict_, f, sort_keys=False)
    with open(log_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)

    # Configure
    device = dict_['device']
    cuda = device != 'cpu'
    init_seeds(2 + rank)
    # # org below
    # with open(opt.data) as f:
    #     data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    # train_path = data_dict['train']
    # test_path = data_dict['val']
    # nc, names = (1, ['item']) if opt.single_cls else (int(data_dict['nc']), data_dict['names'])  # number classes, names
    # assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check
    train_path = dict_['train_path']
    test_path = dict_['test_path'] if dict_['validation_mode']=='test' else dict_['val_path']
    nc = dict_['nclasses']
    names = dict_['names']

    # # org below
    # Model
    # pretrained = weights.endswith('.pt')
    # if pretrained:
    #     with torch_distributed_zero_first(rank):
    #         attempt_download(weights)  # download if not found locally
    #     ckpt = torch.load(weights, map_location=device)  # load checkpoint
    #     model = Darknet(dict_).to(device)  # create
    #     state_dict = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
    #     model.load_state_dict(state_dict, strict=False)
    #     print('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    # else:
    #     model = Darknet(dict_).to(device) # create
    
    pretrained = False
    img_size = dict_['img_size']
    model = Darknet(dict_, (img_size, img_size)).to(device) # create

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            pg2.append(v)  # biases
        elif 'Conv2d.weight' in k:
            pg1.append(v)  # apply weight_decay
        else:
            pg0.append(v)  # all else

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
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # Resume
    start_epoch, best_fitness = 0, 0.0
    # # org below
    # if pretrained:
    #     # Optimizer
    #     if ckpt['optimizer'] is not None:
    #         optimizer.load_state_dict(ckpt['optimizer'])
    #         best_fitness = ckpt['best_fitness']

    #     # Results
    #     if ckpt.get('training_results') is not None:
    #         with open(results_file, 'w') as file:
    #             file.write(ckpt['training_results'])  # write results.txt

    #     # Epochs
    #     start_epoch = ckpt['epoch'] + 1
    #     if epochs < start_epoch:
    #         print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
    #               (weights, ckpt['epoch'], epochs))
    #         epochs += ckpt['epoch']  # finetune additional epochs

    #     del ckpt, state_dict
    
    # Image sizes
    gs = 32 # grid size (max stride)
    # imgsz, imgsz_test = [check_img_size(x, gs) for x in dict_['img_size']]  # verify imgsz are gs-multiples
    
    # # org below
    # # DP mode
    # if cuda and rank == -1 and torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)

    # # SyncBatchNorm
    # if opt.sync_bn and cuda and rank != -1:
    #     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    #     print('Using SyncBatchNorm()')

    # Exponential moving average
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # DDP mode
    if cuda and rank != -1:
        model = DDP(model, device_ids=[dict_['local_rank']], output_device=(dict_['local_rank']))

    # Trainloader
    dataloader, dataset = create_dataloader(train_path, dict_['img_size'], batch_size, gs, hyp=hyp, augment=True,
                                            cache=dict_['cache_images'], rect=dict_['rect'], local_rank=rank,
                                            world_size=dict_['world_size'])
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, dict_['names'], nc - 1)

    # Testloader
    if rank in [-1, 0]:
        ema.updates = start_epoch * nb // accumulate  # set EMA updates ***
        # local_rank is set to -1. Because only the first process is expected to do evaluation.
        testloader = create_dataloader(test_path, dict_['img_size'], batch_size, gs, hyp=hyp, augment=False,
                                       cache=dict_['cache_images'], rect=True, local_rank=-1, world_size=dict_['world_size'])[0]

    # Model parameters
    # dict['cls'] *= nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
    model.nc = dict_['nclasses']  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    model.names = names

    # Class frequency
    if rank in [-1, 0]:
        labels = np.concatenate(dataset.labels, 0)
        c = torch.tensor(labels[:, 0])  # classes
        # cf = torch.bincount(c.long(), minlength=nc) + 1.
        # model._initialize_biases(cf.to(device))
        plot_labels(labels, save_dir=log_dir)
        if tb_writer:
            tb_writer.add_histogram('classes', c, 0)

        # Check anchors
        #if not opt.noautoanchor:
        #    check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)

    # Start training
    t0 = time.time()
    nw = max(3 * nb, 1e3)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    if rank in [0, -1]:
        print('Image sizes %g train, %g test' % (dict_['img_size'], dict_['img_size']))
        print('Using %g dataloader workers' % dataloader.num_workers)
        print('Starting training for %g epochs...' % epochs)
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)
        if dataset.image_weights:
            # Generate indices
            if rank in [-1, 0]:
                w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
                image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
                dataset.indices = random.choices(range(dataset.n), weights=image_weights,
                                                 k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = torch.zeros([dataset.n], dtype=torch.int)
                if rank == 0:
                    indices[:] = torch.from_tensor(dataset.indices, dtype=torch.int)
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        if rank in [-1, 0]:
            print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])

            # Multi-scale
            if dict_['multi_scale']:
                sz = random.randrange(dict_['img_size'] * 0.5, dict_['img_size'] * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Autocast
            with amp.autocast(enabled=cuda):
                # Forward
                pred = model(imgs)

                # Loss
                # # org below
                # loss, loss_items = compute_loss(pred, targets.to(device), model)  # scaled by batch_size
                loss, loss_items = compute_loss(pred, targets.to(device), hyp, dict_)  # scaled by batch_size
                if rank != -1:
                    loss *= dict_['world_size']  # gradient averaged between devices in DDP mode
                # if not torch.isfinite(loss):
                #     print('WARNING: non-finite loss, ending training ', loss_items)
                #     return results

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                # Plot
                if ni < 3:
                    f = str(log_dir / ('train_batch%g.jpg' % ni))  # filename
                    result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                    if tb_writer and result is not None:
                        tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                        # tb_writer.add_graph(model, imgs)  # add model to tensorboard

            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            if ema is not None:
                ema.update_attr(model)
            final_epoch = epoch + 1 == epochs
            if dict_['test_all'] or final_epoch:  # Calculate mAP
                # # org below
                # results, maps, times = test.test(opt.data,
                #                                  batch_size=batch_size,
                #                                  imgsz=imgsz_test,
                #                                  save_json=final_epoch and opt.data.endswith(os.sep + 'coco.yaml'),
                #                                  model=ema.ema.module if hasattr(ema.ema, 'module') else ema.ema,
                #                                  single_cls=opt.single_cls,
                #                                  dataloader=testloader,
                #                                  save_dir=log_dir)

                results, maps, times = test(dict_,
                                            hyp,
                                            model = ema.ema.module if hasattr(ema.ema, 'module') else ema.ema,
                                            augment=False,
                                            dataloader=testloader)


            # Write
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
            # # org below
            # if len(names) and opt.bucket:
            #     os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

            # Tensorboard
            if tb_writer:
                tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                        'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                        'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
                for x, tag in zip(list(mloss[:-1]) + list(results), tags):
                    tb_writer.add_scalar(tag, x, epoch)

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
            if fi > best_fitness:
                best_fitness = fi

            # Save model
            # # org below
            # save = (not opt.nosave) or (final_epoch and not opt.evolve)
            save = (dict_['save_all']!=False) or (final_epoch)
            if save:
                with open(results_file, 'r') as f:  # create checkpoint
                    ckpt = {'epoch': epoch,
                            'best_fitness': best_fitness,
                            'training_results': f.read(),
                            'model': ema.ema.module.state_dict() if hasattr(ema, 'module') else ema.ema.state_dict(),
                            'optimizer': None if final_epoch else optimizer.state_dict()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if epoch >= (epochs-5):
                    torch.save(ckpt, last.replace('.pt','_{:03d}.pt'.format(epoch)))
                if (best_fitness == fi) and not final_epoch:
                    torch.save(ckpt, best)
                del ckpt
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    if rank in [-1, 0]:
        # Strip optimizers
        n = ('_' if len(dict_['names']) and not dict_['names'].isnumeric() else '') + dict_['names']
        fresults, flast, fbest = 'results%s.txt' % n, wdir + 'last%s.pt' % n, wdir + 'best%s.pt' % n
        for f1, f2 in zip([wdir + 'last.pt', wdir + 'best.pt', 'results.txt'], [flast, fbest, fresults]):
            if os.path.exists(f1):
                os.rename(f1, f2)  # rename
                ispt = f2.endswith('.pt')  # is *.pt
                strip_optimizer(f2) if ispt else None  # strip optimizer
                # os.system('gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket and ispt else None  # upload
        # Finish
        # # org below
        # if not opt.evolve:
        #     plot_results(save_dir=log_dir)  # save as results.png
        print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))

    dist.destroy_process_group() if rank not in [-1, 0] else None
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    dict_ = { 
        'device':'cuda', #Intialise device as cpu. Later check if cuda is avaialbel and change to cuda    
        'device_num': '0',

        'anchors_g': [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]],
        'nclasses': 3, #Number of classes
        'names' : ['person', 'bicycle', 'car'],
        'gs': 32, #Image size multiples
        'img_size': 320, #Input image size. Must be a multiple of 32
        'strides': [8,16,32], #strides of p3,p4,p5
        'epochs': 30, #number of epochs
        'batch_size': 8, #train batch size
        'test_size': 8, #test batch size
        'use_adam': False, #Bool to use Adam optimiser
        'use_ema': True, #Exponential moving average control
        'multi_scale': False, #Bool to do multi-scale training
        'gr' : 1.0, # giou loss ratio (obj_loss = 1.0 or giou)
        'nms_conf_t':0.2, # Confidence training threshold
        'nms_merge': False,
        'test_all': True, #Run test after end of each epoch
        'save_all': False, #Save checkpoints after every epoch
        'plot_all': True,

        'global_rank': -1,
        'local_rank': -1,
        'world_size': 1,
        'cache_images': True,
        'rect': True,
        'evolve': False,
        
        #____________________ PATH
        'weight_path': './Fusion/yolo_pre_3c.pt',
        # 'train_path': DATASET_PP_PATH + '/train/RGB_cropped/',
        'train_path': '/data/Sam/FLIR_PP/val/',

        'validation_mode': 'val', # change to test for the final test
        # 'val_path': DATASET_PP_PATH + '/val/',
        'val_path': '/data/Sam/FLIR_PP/val/',

        # 'test_path' : '/home/ub145/Documents/Dataset/FLIR/FLIR_PP/val',
        'test_path' : '/data/Sam/FLIR_PP/val/',

        'save_dir': './save_dir/',
        'logdir': './runs',
        # 'train_img_path': DATASET_PP_PATH + '/train/RGB_cropped/',
        # 'train_label_path': DATASET_PP_PATH + '/train/yolo_format_labels',

        # 'validation_mode': 'validation', # change to test for the final test
        # 'val_img_path': DATASET_PP_PATH + '/val/',
        # 'val_label_path': DATASET_PP_PATH + '/val/yolo_format_labels',

        # 'test_img_path' : DATASET_PP_PATH + '/test/RGB_cropped/',
        # 'test_label_path': DATASET_PP_PATH + '/test/yolo_format_labels',
     }

    hyp = {
        #____________________ Hyp
        'lr0': 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
        'momentum': 0.937,  # SGD momentum/Adam beta1
        'weight_decay': 0.0005,  # optimizer weight decay
        'giou': 0.05,  # GIoU loss gain
        'cls': 0.5,  # cls loss gain 0.025_goku
        'cls_pw': 1.0,  # cls BCELoss positive_weight
        'obj': 1.0,  # obj loss gain (scale with pixels)
        'obj_pw': 1.0,  # obj BCELoss positive_weight
        'iou_t': 0.6,  # IoU training threshold
        'anchor_t': 4.0,  # anchor-multiple threshold
        'fl_gamma': 0.0,  # focal loss gamma (efficientDet default gamma=1.5)
        'hsv_h': 0.015,  # image HSV-Hue augmentation (fraction)
        'hsv_s': 0.7,  # image HSV-Saturation augmentation (fraction)
        'hsv_v': 0.4,  # image HSV-Value augmentation (fraction)
        'degrees': 0.0,  # image rotation (+/- deg)
        'translate': 0.0,  # image translation (+/- fraction)
        'scale': 0.5,  # image scale (+/- gain)
        'shear': 0.0,  # image shear (+/- deg)
        'perspective': 0.0,  # image perspective (+/- fraction), range 0-0.001
        'flipud': 0.0,  # image flip up-down (probability)
        'fliplr': 0.5,  # image flip left-right (probability)
        'mixup': 0.0, #mix up probability
    }

    # # Resume
    # if opt.resume:
    #     last = get_latest_run() if opt.resume == 'get_last' else opt.resume  # resume from most recent run
    #     if last and not opt.weights:
    #         print(f'Resuming training from {last}')
    #     opt.weights = last if opt.resume and not opt.weights else opt.weights
    # if opt.local_rank == -1 or ("RANK" in os.environ and os.environ["RANK"] == "0"):
    #     check_git_status()

    # opt.hyp = opt.hyp or ('data/hyp.scratch.yaml')
    # opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
    # assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'

    # opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    # device = select_device(opt.device, batch_size=opt.batch_size)
    # opt.total_batch_size = opt.batch_size
    # opt.world_size = 1
    # opt.global_rank = -1

    # # DDP mode
    # if opt.local_rank != -1:
    #     assert torch.cuda.device_count() > opt.local_rank
    #     torch.cuda.set_device(opt.local_rank)
    #     device = torch.device('cuda', opt.local_rank)
    #     dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
    #     opt.world_size = dist.get_world_size()
    #     opt.global_rank = dist.get_rank()
    #     assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
    #     opt.batch_size = opt.total_batch_size // opt.world_size

    # print(opt)
    # with open(opt.hyp) as f:
    #     hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps

    # Train
    if not dict_['evolve']:
        tb_writer = None
        if dict_['global_rank'] in [-1, 0]:
            print('Start Tensorboard with "tensorboard --logdir %s", view at http://localhost:6006/' % dict_['logdir'])
            tb_writer = SummaryWriter(log_dir=increment_dir(Path(dict_['logdir']) / 'exp', dict_['device']))  # runs/exp

        train(dict_, hyp, tb_writer)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'momentum': (0.1, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'giou': (1, 0.02, 0.2),  # GIoU loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (1, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (0, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (1, 0.0, 1.0),  # image flip left-right (probability)
                'mixup': (1, 0.0, 1.0)}  # image mixup (probability)

        assert dict_['local_rank'] == -1, 'DDP mode not implemented for --evolve'
        # # org below
        # opt.notest, opt.nosave = True, True  # only test/save final epoch
        # # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        # yaml_file = Path('runs/evolve/hyp_evolved.yaml')  # save best result here
        # if opt.bucket:
        #     os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        for _ in range(100):  # generations to evolve
            if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.9, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy())

            # Write mutation results
            # print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # Plot results
        # plot_evolution(yaml_file)
        # print('Hyperparameter evolution complete. Best results saved as: %s\nCommand to train a new model with these '
        #       'hyperparameters: $ python train.py --hyp %s' % (yaml_file, yaml_file))