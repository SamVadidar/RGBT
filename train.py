# -*- coding: utf-8 -*-
"""
Adapted From: WongKinYiu and Gokulesh Danapal
https://github.com/WongKinYiu/ScaledYOLOv4
https://github.com/gokulesh-danapal
"""
import os
import math
import time
import yaml
import torch
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from torch.cuda import amp
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from test import test
from Fusion.yolo import Darknet
from Fusion.data_processor import Dataset
from FLIR_PP.arg_parser import DATASET_PP_PATH, DATASET_PATH
from Fusion.utils import init_seeds, init_seeds_master, increment_dir, box_iou, fitness

    
class FocalLoss(torch.nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps

def compute_loss(p, targets, hyp):  # predictions, targets, model
    device = targets.device
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
    tcls, tbox, indices, anchors = build_targets(p, targets, hyp)  # targets
    h = hyp  # hyperparameters
    
    # Define criteria
    BCEcls = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['cls_pw']])).to(device)
    BCEobj = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['obj_pw']])).to(device)
    
    # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = smooth_BCE(eps=0.0)
    
    # Focal loss
    g = h['fl_gamma']  # focal loss gamma
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
    
    # Losses
    nt = 0  # number of targets
    np = len(p)  # number of outputs
    balance = [4.0, 1.0, 0.4] if np == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
    for i, pi in enumerate(p): # layer index, layer predictions
        #pi = pi.detach()
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
    
        n = b.shape[0]  # number of targets
        if n:
            nt += n  # cumulative targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
    
            # Regression
            pxy = ps[:, :2].sigmoid() * 2. - 0.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
            #pxy = torch.sigmoid(ps[:, 0:2])  # pxy = pxy * s - (s - 1) / 2,  s = 1.5  (scale_xy)
            #pwh = torch.exp(ps[:, 2:4]).clamp(max=1E3) * anchors[i]
            pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
            giou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # giou(prediction, target)
            lbox += (1.0 - giou).mean()  # giou loss
    
            # Objectness
            tobj[b, a, gj, gi] = (1.0 - hyp['gr']) + hyp['gr'] * giou.detach().clamp(0).type(tobj.dtype)  # giou ratio
    
            # Classification
            if hyp['nclasses'] > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                t[range(n), tcls[i]] = cp
                lcls += BCEcls(ps[:, 5:], t)  # BCE
    
            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
    
        lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss
    
    s = 3 / np  # output count scaling
    lbox *= h['giou'] * s
    lobj *= h['obj'] * s * (1.4 if np == 4 else 1.)
    lcls *= h['cls'] * s
    bs = tobj.shape[0]  # batch size
    
    loss = lbox + lobj + lcls
    return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

def build_targets(p, targets, hyp):
    nt = targets.shape[0]  # number of anchors, targets
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain
    off = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float()  # overlap offsets
    
    g = 0.5  # offset
    for i, ancs in enumerate(torch.from_numpy(np.array(hyp['anchors_g'])).reshape(3,3,2).to(hyp['device'])):#model.module.yolo_layers if multi_gpu else model.yolo_layers):
        # get number of grid points and anchor vec for this yolo layer
        #anchors = model.module.module_list[jj].anchor_vec if multi_gpu else model.module_list[jj].anchor_vec
        ancs = ancs // hyp['strides'][i]
        gain[2:] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
    
        # Match targets to anchors
        a, t, offsets = [], targets * gain, 0
        if nt:
            na = ancs.shape[0]  # number of anchors
            at = torch.arange(na).view(na, 1).repeat(1, nt)  # anchor tensor, same as .repeat_interleave(nt)
            r = t[None, :, 4:6] / ancs[:, None]  # wh ratio
            j = torch.max(r, 1. / r).max(2)[0] < hyp['anchor_t']# compare
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
            a, t = at[j], t.repeat(na, 1, 1)[j]  # filter
    
            # overlaps
            gxy = t[:, 2:4]  # grid xy
            z = torch.zeros_like(gxy)
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxy % 1. > (1 - g)) & (gxy < (gain[[2, 3]] - 1.))).T
            a, t = torch.cat((a, a[j], a[k], a[l], a[m]), 0), torch.cat((t, t[j], t[k], t[l], t[m]), 0)
            offsets = torch.cat((z, z[j] + off[0], z[k] + off[1], z[l] + off[2], z[m] + off[3]), 0) * g
    
        # Define
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices
    
        # Append
        #indices.append((b, a, gj, gi))  # image, anchor, grid indices
        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(ancs[a])  # anchors
        tcls.append(c)  # class
        
    return tcls, tbox, indices, anch

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v + 1e-16)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou

def is_parallel(model):
    return type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)

def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)

def train(hyp, tb_writer, dataset, ckpt_path= None, test_set = None):
    log_dir = Path(tb_writer.log_dir) # logging directory
    wdir = os.path.join(log_dir,'weights') + os.sep  # weights directory
    os.makedirs(wdir, exist_ok=True)
    last = wdir + 'last.pt'
    best = wdir + 'best.pt'
    results_file = os.path.join(log_dir,'results.txt')
    #save hyperparameter settings
    with open(log_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    # Configure
    init_seeds_master(1)
    # Model
    model = Darknet(nclasses=hyp['nclasses'], anchors=np.array(hyp['anchors_g'])).to(hyp['device'])
    print('built model')
    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / hyp['batch_size']), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= hyp['batch_size'] * accumulate / nbs  # scale weight_decay

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            pg2.append(v)  # biases
        elif 'conv.weight' in k: # or '1.weight'in k:
            pg1.append(v)  # apply weight_decay
        elif '1.weight' in k:
            pg1.append(v)
        else:
            pg0.append(v)  # all else

    if hyp['use_adam']:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    lf = lambda x: (((1 + math.cos(x * math.pi / hyp['epochs'])) / 2) ** 1.0) * 0.8 + 0.2  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']
            
        # Results
        if ckpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(ckpt['training_results'])  # write results.txt
    
        # Epochs
        if ckpt['epoch'] is not None:
            start_epoch = ckpt['epoch'] + 1
            if hyp['epochs'] < start_epoch:
                print('Model has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                      (ckpt['epoch'], hyp['epochs']))
                hyp['epochs'] += ckpt['epoch']  # finetune additional epochs
    
        del ckpt
    
    # Exponential moving average
    ema = ModelEMA(model)
    
    # Trainloader
    dataloader = torch.utils.data.DataLoader(dataset=dataset,batch_size=hyp['batch_size'],collate_fn=Dataset.collate_fn,shuffle=True)
    nb = len(dataloader)  # number of batches
    # Start training
    t0 = time.time()
    nw = max(3 * nb, 1e3)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(hyp['nclasses'])  # mAP per class
    results = [0, 0, 0, 0, 0, 0]  # 'P', 'R', 'mAP', 'val GIoU', 'val Objectness', 'val Classification'
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(hyp['device'] =='cuda')
    print('Starting training for %g epochs...' % hyp['epochs'])
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, hyp['epochs']):  # epoch ------------------------------------------------------------------
        model.train()
        mloss = torch.zeros(4, device=hyp['device'])  # mean losses
        pbar = enumerate(dataloader)
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
        pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, _, paths) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(hyp['device'], non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
    
            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / hyp['batch_size']]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])
    
            # Multi-scale
            if hyp['multi_scale']:
                sz = random.randrange(hyp['img_size'] * 0.5, hyp['img_size'] * 1.5 + hyp['gs']) // hyp['gs'] * hyp['gs']  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / hyp['gs']) * hyp['gs'] for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
    
            # Autocast
            with amp.autocast(enabled = hyp['device'] =='cuda'):
                # Forward
                pred = model(imgs)
                # Loss
                loss, loss_items = compute_loss(pred, targets.to(hyp['device']),hyp)  # scaled by batch_size
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
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 6) % (
                '%g/%g' % (epoch, hyp['epochs'] - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
            pbar.set_description(s)
            # end batch ------------------------------------------------------------------------------------------------
        # Scheduler
        scheduler.step()
    
        if ema is not None:
            ema.update_attr(model)
        final_epoch = epoch + 1 == hyp['epochs']

        if hyp['test_all'] or final_epoch:  # Calculate mAP
            test_results, maps, times = test(test_set, hyp, model)
            results[:3] = test_results[:3]; results[3:] = loss_items[:3].cpu(); 
        
        # Write
        with open(results_file, 'a') as f:
            f.write('%s'*6 % tuple(np.array(results).astype('str')) + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
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
        if hyp['save_all'] or final_epoch:
            with open(results_file, 'r') as f:  # create checkpoint
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': f.read(),
                        'model': ema.ema.module.state_dict() if hasattr(ema, 'module') else ema.ema.state_dict(),
                        'optimizer': None if final_epoch else optimizer.state_dict()}
    
            # Save last, best and delete
            torch.save(ckpt, last)
            if epoch >= (hyp['epochs']-5):
                torch.save(ckpt, last.replace('.pt','_{:03d}.pt'.format(epoch)))
            if (best_fitness == fi) and not final_epoch:
                torch.save(ckpt, best)
            del ckpt
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
        # Finish
    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    torch.cuda.empty_cache()
    return results

if __name__ == '__main__':
    hyp = { 'device':'cuda', #Intialise device as cpu. Later check if cuda is avaialbel and change to cuda    
            'lr0': 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
            'momentum': 0.937,  # SGD momentum/Adam beta1
            'weight_decay': 0.0005,  # optimizer weight decay
            'anchors_g': [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]],
            'nclasses': 3, #Number of classes
            'names' :  ['person', 'bicycle', 'car'],
            'gs': 32, #Image size multiples
            'img_size': 256, #Input image size. Must be a multiple of 32
            'strides': [8,16,32], #strides of p3,p4,p5
            'epochs': 30, #number of epochs
            'batch_size': 1, #train batch size
            # 'test_size': 16, #test batch size
            'use_adam': False, #Bool to use Adam optimiser
            'use_ema': True, #Exponential moving average control
            'multi_scale': False, #Bool to do multi-scale training
            'test_all': True, #Run test after end of each epoch
            'save_all': True, #Save checkpoints after every epoch
            
            'giou': 0.05,  # GIoU loss gain
            'cls': 0.025,  # cls loss gain
            'cls_pw': 1.0,  # cls BCELoss positive_weight
            'obj': 1.0,  # obj loss gain (scale with pixels)
            'obj_pw': 1.0,  # obj BCELoss positive_weight
            'gr' : 1.0, # giou loss ratio (obj_loss = 1.0 or giou)
            'iou_t': 0.6,  # IoU training threshold
            'conf_t':0.2, # Confidence training threshold
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
            'mixup': 0.0 #mix up probability
        }

    TRAIN_SET_IMG_PATH = DATASET_PP_PATH + '/train/RGB_cropped/'
    TRAIN_SET_LABEL_PATH = DATASET_PP_PATH + '/train/yolo_format_labels'

    VAL_SET_IMG_PATH = DATASET_PP_PATH + '/val/RGB_cropped/'
    VAL_SET_LABEL_PATH = DATASET_PP_PATH + '/val/yolo_format_labels'
    # VAL_SET_IMG_PATH = DATASET_PATH + '/val/thermal_8_bit'

    LOG_DIR = './Fusion/runs'
    WEIGHT_PATH = './Fusion/yolo_pre_3c.pt'

    train_set = Dataset(hyp, TRAIN_SET_IMG_PATH, TRAIN_SET_LABEL_PATH, augment= True)
    test_set = Dataset(hyp, VAL_SET_IMG_PATH, VAL_SET_LABEL_PATH, augment= False, mosaic=False)

    tb_writer = SummaryWriter(log_dir = LOG_DIR)
    results = train(hyp, tb_writer, train_set, WEIGHT_PATH, test_set)