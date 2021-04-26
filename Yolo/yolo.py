# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 14:26:51 2021

@author: Gokulesh Danapal (GX6)

Confidentiality: Internal
"""

import torch
import torchvision as tv
import torch.nn.functional as F
import math
import numpy as np
from PIL import Image
import os
import random
import cv2
from copy import deepcopy
import time
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from pathlib import Path
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda import amp
from tqdm import tqdm
import time
# %matplotlib inline

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
use_adam = False
multi_scale = False
anchors_g = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
anchors_g = np.array(anchors_g).reshape(-1,2)
strides = [8,16,32]
epochs = 300
batch_size = 4
total_batch_size = 4
rank = -1
# names = ['person', 'bicycle', 'car']
names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']
hyp = {'lr0': 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'momentum': 0.937,  # SGD momentum/Adam beta1
                'weight_decay': 0.0005,  # optimizer weight decay
                'giou': 0.05,  # GIoU loss gain
                'cls': 0.5,  # cls loss gain
                'cls_pw': 1.0,  # cls BCELoss positive_weight
                'obj': 1.0,  # obj loss gain (scale with pixels)
                'obj_pw': 1.0,  # obj BCELoss positive_weight
                'iou_t': 0.6,  # IoU training threshold
                'conf_t':0.5, # Confidence training threshold
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
                'fliplr': 0.0,  # image flip left-right (probability)
                'mixup': 0.0}
nc = 80 #number of classes
gr = 1.0 # giou loss ratio (obj_loss = 1.0 or giou)
img_size = 640
mosaic_border = [-img_size // 2, -img_size // 2]
tensor = tv.transforms.ToTensor()
image = tv.transforms.ToPILImage()
#%%
weight_path = "./Yolo/yolov4-csp_pretrained.pytorch"
imroot = "/home/ub145/Documents/Dataset/FLIR/FLIR/FLIR_PP/train/RGB_cropped"
#rroot =
lroot = "/home/ub145/Documents/Dataset/FLIR/FLIR/FLIR_PP/train/yolo_format_labels"
inputs = list(os.listdir(imroot))
#%%


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls: Predicted object classes (nparray).
        target_cls: True object classes (nparray).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    pr_score = 0.1  # score to evaluate P and R https://github.com/ultralytics/yolov3/issues/898
    s = [unique_classes.shape[0], tp.shape[1]]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 or n_gt == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_gt + 1e-16)  # recall curve
            r[ci] = np.interp(-pr_score, -conf[i], recall[:, 0])  # r at pr_score, negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-pr_score, -conf[i], precision[:, 0])  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

            # Plot
            # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            # ax.plot(recall, precision)
            # ax.set_xlabel('Recall')
            # ax.set_ylabel('Precision')
            # ax.set_xlim(0, 1.01)
            # ax.set_ylim(0, 1.01)
            # fig.tight_layout()
            # fig.savefig('PR_curve.png', dpi=300)

    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def labels_to_class_weights(labels, nc=80):
    # Get class weights (inverse frequency) from training labels
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurences per class

    # Prepend gridpoint count (for uCE trianing)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights)

def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    init_seeds(seed=seed)

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

def compute_loss(p, targets):  # predictions, targets, model
    device = targets.device
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
    tcls, tbox, indices, anchors = build_targets(p, targets)  # targets
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
        pi = pi.detach()
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
            tobj[b, a, gj, gi] = (1.0 - gr) + gr * giou.detach().clamp(0).type(tobj.dtype)  # giou ratio

            # Classification
            if nc > 1:  # cls loss (only if multiple classes)
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

def build_targets(p, targets):
    nt = targets.shape[0]  # number of anchors, targets
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain
    off = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float()  # overlap offsets

    g = 0.5  # offset
    for i, ancs in enumerate(anchors_g.reshape(3,3,2)):#model.module.yolo_layers if multi_gpu else model.yolo_layers):
        # get number of grid points and anchor vec for this yolo layer
        #anchors = model.module.module_list[jj].anchor_vec if multi_gpu else model.module_list[jj].anchor_vec
        ancs = ancs // strides[i]
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

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates
    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        classes = 3
        # Filter by class
        if classes:
            x = x[(x[:, 5:6] < torch.tensor(classes, device=x.device)).any(1)]
            # x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = tv.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    # imges = []
    # img_boxes = []
    # new_output = []
    # for img in output.cpu():
    #     for box in img:
    #         # the last element in the box refers to class number
    #         if box[-1] < 3:
    #             img_boxes.append(box)
    #     imges.append(img_boxes)

    # new_output = torch.from_numpy(np.array(imges))

    return output

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

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def load_image(index):
    path = os.path.join(imroot, inputs[index])
    img = cv2.imread(path)
    h0, w0 = img.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        #interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
    return img, (h0, w0), img.shape[:2]

def load_radar(index):
    path = os.path.join(imroot, inputs[index])
    img = Image.open(path)
    return img, img.shape[1], img.shape[0]

def load_label(index):
    path = os.path.join(lroot, inputs[index].replace('.jpg', '.txt'))
    try:
        with open(path, 'r') as f:
            l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)  # labels
            return l
    except:
        print(path)
        #if len(l) == 0:
            #l = np.zeros((0, 5), dtype=np.float32)
            #print('Zero label')

def random_perspective(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2
    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)
    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)
    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)
    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)
    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)
        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, 1:5] = xy[i]
    return img, targets

def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates

def create_mosaic(index):
    labels4 = []
    s = img_size
    yc, xc = s, s  # mosaic center x, y
    indices = [index] + [random.randint(0, len(inputs) - 1) for _ in range(3)]  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        x = load_label(index)
        labels = x.copy()
        if x.size > 0:  # Normalized xywh to pixel xyxy format
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
        labels4.append(labels)

    # Concat/clip labels
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])
    #img4, labels4 = random_perspective(img4, labels4, degrees=hyp['degrees'],translate=hyp['translate'],scale=hyp['scale'],shear=hyp['shear'],perspective=hyp['perspective'],border=mosaic_border)
    return img4, labels4

class Dataset(object):
  def __init__(self,imroot,lroot,augment = True):
    self.augment = augment
    self.imroot = imroot
    #self.rroot = rroot
    self.lroot = lroot
    self.inputs = list(os.listdir(imroot))
    self.fliplr = tv.transforms.RandomHorizontalFlip(p=1)
    self.flipud = tv.transforms.RandomVerticalFlip(p=1)
  def __getitem__(self,index):
    # load images
    shapes = None
    if self.augment:
        img, labels = create_mosaic(index)
         #MixUp https://arxiv.org/pdf/1710.09412.pdf
        if random.random() < hyp['mixup']:
            img2, labels2 = create_mosaic(random.randint(0, len(labels) - 1))
            r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
            img = (img * r + img2 * (1 - r)).astype(np.uint8)
            labels = np.concatenate((labels, labels2), 0)
        nL = len(labels)  # number of labels
        #flip up-down
        #if random.random() < hyp['flipud']:
            #img = np.flipud(img)
            #labels[:, 2] = img.shape[1] - labels[:, 2]
        # flip left-right
        #if random.random() < hyp['fliplr']:
          #img = np.fliplr(img)
          #labels[:, 2] = img.shape[1] - labels[:, 2]
    else:
        # Load image
        img, (h0, w0), (h, w) = load_image(index)
        # Letterbox
        img, ratio, pad = letterbox(img, img_size, auto=False, scaleup=self.augment)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
        # Load labels
        x = load_label(index)
        labels = x.copy()
        nL = len(x)
        if nL:
            labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
            labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
            labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
            labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]
    if nL:
        labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
        labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
        labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1
    if self.augment:
        #flip up-down
        if random.random() < hyp['flipud']:
            img = np.flipud(img)
            labels[:, 2] = 1 - labels[:, 2]
        # flip left-right
        if random.random() < hyp['fliplr']:
          img = np.fliplr(img)
          labels[:, 1] = 1 - labels[:, 1]
    labels_out = torch.zeros((len(labels), 6))
    if nL:
        labels_out[:, 1:] = torch.from_numpy(labels)
    # Convert
    img = img.copy()
    img = img.transpose(2, 0, 1)  # BGR to RGB, to 3x416x416 [:, :, ::-1]
    #img = np.ascontiguousarray(img)
    return torch.from_numpy(img), labels_out, shapes, inputs[index]

  @staticmethod
  def collate_fn(batch):
    img, label,shapes,paths  = zip(*batch)  # transposed
    for i, l in enumerate(label):
        l[:, 0] = i  # add target image index for build_targets()
    return torch.stack(img, 0), torch.cat(label, 0), shapes, paths

  def __len__(self):
    return len(self.inputs)

def scale_img(img, ratio=1.0, same_shape=False):  # img(16,3,256,416), r=ratio
    # scales img(bs,3,y,x) by ratio
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # new size
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
        if not same_shape:  # pad/crop img
            gs = 32  # (pixels) grid size
            h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean

class Mish(torch.nn.Module):
    def __init__(self):
        super(Mish,self).__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x

class CBM(torch.nn.Module):
    def __init__(self,in_filters, out_filters, kernel_size, stride):
        super(CBM,self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=in_filters,out_channels=out_filters,kernel_size=kernel_size,stride=stride,padding=kernel_size//2,bias=False)
        self.batchnorm = torch.nn.BatchNorm2d(num_features=out_filters,momentum=0.03, eps=1E-4)
        self.act = Mish()
    def forward(self,x):
        return self.act(self.batchnorm(self.conv(x)))

class ResUnit(torch.nn.Module):
    def __init__(self, filters, first = False):
        super(ResUnit, self).__init__()
        if first:
            self.out_filters = filters//2
        else:
            self.out_filters = filters
        self.resroute= torch.nn.Sequential(CBM(filters, self.out_filters, kernel_size=1, stride=1),
                                                    CBM(self.out_filters, filters, kernel_size=3, stride=1))
    def forward(self, x):
        shortcut = x
        x = self.resroute(x)
        return x+shortcut

class CSP(torch.nn.Module):
    def __init__(self, filters, nblocks):
        super(CSP,self).__init__()
        self.skip = CBM(in_filters=filters,out_filters=filters//2,kernel_size=1,stride=1)
        self.route_list = torch.nn.ModuleList()
        self.route_list.append(CBM(in_filters=filters,out_filters=filters//2,kernel_size=1,stride=1))
        for block in range(nblocks):
            self.route_list.append(ResUnit(filters=filters//2))
        self.route_list.append(CBM(in_filters=filters//2,out_filters=filters//2,kernel_size=1,stride=1))
        self.last = CBM(in_filters=filters,out_filters=filters,kernel_size=1,stride=1)

    def forward(self,x):
        shortcut = self.skip(x)
        for block in self.route_list:
            x = block(x)
        x = torch.cat((x,shortcut),dim = 1)
        return self.last(x)

class SPP(torch.nn.Module):
    def __init__(self,filters):
        super(SPP,self).__init__()
        self.maxpool5 = torch.nn.MaxPool2d(kernel_size=5,stride=1,padding = 5//2)
        self.maxpool9 = torch.nn.MaxPool2d(kernel_size=9,stride=1,padding = 9//2)
        self.maxpool13 = torch.nn.MaxPool2d(kernel_size=13,stride=1,padding = 13//2)
    def forward(self,x):
        x5 = self.maxpool5(x)
        x9 = self.maxpool9(x)
        x13 = self.maxpool13(x)
        return torch.cat((x13,x9,x5,x),dim=1)

class rCSP(torch.nn.Module):
    def __init__(self,filters,spp_block = False):
        super(rCSP,self).__init__()
        self.include_spp = spp_block
        if self.include_spp:
            self.in_filters = filters*2
        else:
            self.in_filters = filters
        self.skip = CBM(in_filters=self.in_filters,out_filters=filters,kernel_size=1,stride=1)
        self.module_list = torch.nn.ModuleList()
        self.module_list.append(torch.nn.Sequential(CBM(in_filters=self.in_filters,out_filters=filters,kernel_size=1,stride=1),
                                CBM(in_filters=filters,out_filters=filters,kernel_size=3,stride=1),
                                CBM(in_filters=filters,out_filters=filters,kernel_size=1,stride=1)))
        if self.include_spp:
            self.module_list.append(torch.nn.Sequential(SPP(filters=filters),
                                    CBM(in_filters=filters*4,out_filters=filters,kernel_size=1,stride=1)))
        self.module_list.append(CBM(in_filters=filters,out_filters=filters,kernel_size=3,stride=1))
        self.last = CBM(in_filters=filters*2,out_filters=filters,kernel_size=1,stride=1)
    def forward(self,x):
        shortcut = self.skip(x)
        for block in self.module_list:
            x = block(x)
        x = torch.cat((x,shortcut),dim=1)
        x = self.last(x)
        return x

def up(filters):
        return torch.nn.Sequential(CBM(in_filters=filters,out_filters=filters//2,kernel_size=1,stride=1),
                                        torch.nn.Upsample(scale_factor=2))

class YOLOLayer(torch.nn.Module):
    def __init__(self, anchors, nc, stride):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        #self.index = yolo_index  # index of this layer in layers
        #self.layers = layers  # model output layer indices
        self.stride = stride  # layer stride
        #self.nl = len(layers)  # number of output layers (3)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs (85)
        self.nx, self.ny, self.ng = 0, 0, 0  # initialize number of x, y gridpoints
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)

    def create_grids(self, ng=(13, 13), device='cpu'):
        self.nx, self.ny = ng  # x and y grid size
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offsets
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)])
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p):
        bs, _, ny, nx = p.shape  # bs, 255, 13, 13
        if (self.nx, self.ny) != (nx, ny):
            self.create_grids((nx, ny), p.device)
        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p
        else:  # inference
            io = p.sigmoid()
            io[..., :2] = (io[..., :2] * 2. - 0.5 + self.grid)
            io[..., 2:4] = (io[..., 2:4] * 2) ** 2 * self.anchor_wh
            io[..., :4] *= self.stride
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]


class Backbone(torch.nn.Module):
    def __init__(self):
        super(Backbone,self).__init__()
        self.main3 = torch.nn.Sequential(CBM(in_filters=3,out_filters=32,kernel_size=3,stride=1),
                                        CBM(in_filters=32,out_filters=64,kernel_size=3,stride=2),
                                        ResUnit(filters = 64, first= True),
                                        CBM(in_filters=64,out_filters=128,kernel_size=3,stride=2),
                                        CSP(filters=128,nblocks = 2),
                                        CBM(in_filters=128,out_filters=256,kernel_size=3,stride=2),
                                        CSP(filters=256,nblocks = 8))
        self.main4 = torch.nn.Sequential(CBM(in_filters=256,out_filters=512,kernel_size=3,stride=2),
                                        CSP(filters=512,nblocks = 8))
        self.main5 = torch.nn.Sequential(CBM(in_filters=512,out_filters=1024,kernel_size=3,stride=2),
                                        CSP(filters=1024,nblocks = 4))
    def forward(self,x):
        x3 = self.main3(x)
        x4 = self.main4(x3)
        x5 = self.main5(x4)
        return (x3,x4,x5)

class Neck(torch.nn.Module):
    def __init__(self):
        super(Neck,self).__init__()
        self.main5 = rCSP(512,spp_block=True)
        self.up5 = up(512)
        self.conv1 = CBM(in_filters=512,out_filters=256,kernel_size=1,stride=1)
        self.conv2 = CBM(in_filters=512,out_filters=256,kernel_size=1,stride=1)
        self.main4 = rCSP(256)
        self.up4 = up(256)
        self.conv3 = CBM(in_filters=256,out_filters=128,kernel_size=1,stride=1)
        self.conv4 = CBM(in_filters=256,out_filters=128,kernel_size=1,stride=1)
        self.main3 = rCSP(128)
    def forward(self,x):
        x3 = x[0]; x4 = x[1]; x5= x[2];
        x5 = self.main5(x5)
        x4 = self.main4(self.conv2(torch.cat((self.conv1(x4),self.up5(x5)),dim=1)))
        x3 = self.main3(self.conv4(torch.cat((self.conv3(x3),self.up4(x4)),dim=1)))
        return (x3,x4,x5)

class Head(torch.nn.Module):
    def __init__(self,nclasses):
        super(Head,self).__init__()
        self.last_layers = 3*(4+1+nclasses)
        self.last3 = torch.nn.Sequential(CBM(in_filters=128,out_filters=256,kernel_size=3,stride=1),
                                         torch.nn.Conv2d(in_channels=256,out_channels=self.last_layers,kernel_size=1,stride=1,bias=True))
        self.conv1 = CBM(in_filters=128,out_filters=256,kernel_size=3,stride=2)
        self.conv2 = CBM(in_filters=512,out_filters=256,kernel_size=1,stride=1)
        self.main4 = rCSP(256)
        self.last4 = torch.nn.Sequential(CBM(in_filters=256,out_filters=512,kernel_size=3,stride=1),
                                         torch.nn.Conv2d(in_channels=512,out_channels=self.last_layers,kernel_size=1,stride=1,bias=True))
        self.conv3 = CBM(in_filters=256,out_filters=512,kernel_size=3,stride=2)
        self.conv4 = CBM(in_filters=1024,out_filters=512,kernel_size=1,stride=1)
        self.main5 = rCSP(512)
        self.last5 = torch.nn.Sequential(CBM(in_filters=512,out_filters=1024,kernel_size=3,stride=1),
                                         torch.nn.Conv2d(in_channels=1024,out_channels=self.last_layers,kernel_size=1,stride=1,bias=True))
    def forward(self,x):
        x3 = x[0]; x4 = x[1]; x5= x[2];
        y3 = self.last3(x3)
        x4 = self.main4(self.conv2(torch.cat((self.conv1(x3),x4),dim=1)))
        y4 = self.last4(x4)
        x5 = self.main5(self.conv4(torch.cat((self.conv3(x4),x5),dim=1)))
        y5 = self.last5(x5)
        return y3,y4,y5

class Darknet(torch.nn.Module):
    def __init__(self,nclasses,anchors):
        super(Darknet,self).__init__()
        self.nclasses = nclasses
        self.anchors = anchors
        self.backbone = Backbone()
        self.neck = Neck()
        self.head = Head(self.nclasses)
        self.yolo3 = YOLOLayer(self.anchors[0:3], self.nclasses, stride = 8)
        self.yolo4 = YOLOLayer(self.anchors[3:6], self.nclasses, stride = 16)
        self.yolo5 = YOLOLayer(self.anchors[6:9], self.nclasses, stride = 32)
    def forward(self,x):
        y3,y4,y5 = self.head(self.neck(self.backbone(x)))
        y3 = self.yolo3(y3)
        y4 = self.yolo4(y4)
        y5 = self.yolo5(y5)
        yolo_out = [y3,y4,y5]
        if self.training:
            return yolo_out
        else:  # inference or test
            x, p = zip(*yolo_out)  # inference output, training output
            x = torch.cat(x, 1)  # cat yolo outputs
            return x, p

def train(tb_writer,weights=None):
    log_dir = Path(tb_writer.log_dir) # logging directory
    wdir = os.path.join(log_dir,'weights') + os.sep  # weights directory
    os.makedirs(wdir, exist_ok=True)
    last = wdir + 'last.pytorch'
    best = wdir + 'best.pytorch'
    results_file = os.path.join(log_dir,'results.txt')
    # Configure
    init_seeds(2)
    # Model
    model = Darknet(nclasses=nc, anchors=anchors_g, train= False).to(device)
    if weights is not None:
        model.load_state_dict(torch.load(weights))
    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            pg2.append(v)  # biases
        elif 'Conv.weight' or '1.weight'in k:
            pg1.append(v)  # apply weight_decay
        else:
            pg0.append(v)  # all else

    if use_adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

def test(txt_root = None, plot_all = True, break_no = 1000000):
    #Dataloader
    test_set = Dataset(imroot,lroot,augment=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=8,collate_fn=Dataset.collate_fn,shuffle=True)
    model = Darknet(nclasses=nc, anchors=anchors_g)
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    model = model.to(device)
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    p, r, f1, mp, mr, map50, m_ap, t0, t1, seen = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    stats, ap, ap_class = [], [], []
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    for batch_i, (img, targets, shapes, paths) in enumerate(tqdm(test_loader, desc=s)):
        if batch_i > break_no:
            break
        img = img.to(device)/255.0; targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)
        with torch.no_grad():
            #Run Model
            t = time_synchronized()
            inf_out, train_out = model(img)  # inference and training outputs
            # inf_out = inf_out[:, :, :8]
            t0 += time_synchronized() - t
            #Run NMS
            t = time_synchronized()
            output = non_max_suppression(inf_out, conf_thres=hyp['conf_t'], iou_thres=hyp['iou_t'])
            t1 += time_synchronized() - t
        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue#
            # Append to text file
            if txt_root is not None:
                txt_path = os.path.join(txt_root, paths[si].replace('.jpg', '.txt'))
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                values = pred.clone()
                values[:, :4] = scale_coords(img[si].shape[1:], values[:, :4], shapes[si][0], shapes[si][1])  # to original
                for *xyxy, conf, clas in values:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    with open(txt_path, 'a') as f:
                        f.write(('%g ' * 5 + '\n') % (clas, *xywh))  # label format
            # Clip boxes to image bounds
            clip_coords(pred, (height, width))
            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d not in detected:
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break
            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        if plot_all or batch_i <1:
            boxes = pred
            if pred.type() != None:
                boxes[:, :4] = scale_coords(img[si].shape[1:], boxes[:, :4], shapes[si][0], shapes[si][1])  # to original
                original_img = Image.open(os.path.join(imroot,paths[si]))
                plt.rcParams['figure.figsize'] = (20,20)
                fig,ax = plt.subplots(1)
                ax.imshow(original_img)

                if len(pred):
                    for i, box in enumerate(boxes.cpu()):
                        xmin = box[0]
                        ymin = box[1]
                        w = (box[2]-box[0])
                        h = (box[3]-box[1])
                        rect = patches.Rectangle((xmin,ymin),w,h,linewidth=2,edgecolor='r',facecolor='none')
                        ax.add_patch(rect)
                        ax.text(xmin, ymin, (names[int(box[-1])],int(box[-2]*100)), fontsize = 12)

                plt.show()


    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, m_ap = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, m_ap))

    # Return results
    model.float()  # for training
    maps = np.zeros(nc) + m_ap
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, m_ap), maps, t

#%% try test
import time
start_time = time.time()
(mp, mr, map50, m_ap), maps, t = test(txt_root='./results')
end_time = time.time()
print(f"run time: {end_time - start_time}")
#%% Plot Model
# from torchsummary import summary
# #from pytorch_model_summary import summary
# model = Darknet(nclasses=nc, anchors=anchors_g, train= False)
# torch.save(model.state_dict(), r"C:\Users\TK6YNZ7\Desktop\codes\WorkRep\trunk\yolov4\network_self.pytorch")
# with open(r"C:\Users\TK6YNZ7\Desktop\codes\WorkRep\trunk\yolov4\summary_eval.txt", "w", encoding='utf-8') as f:
#     #print(summary(model, torch.zeros((2,3,416,416)),depth = 10),file = f)
#     print(model.eval(),file = f)
# out = model(torch.zeros((2,3,416,416)))

# #%% Load pretrained weights
# model.load_state_dict(torch.load(r"C:\Users\TK6YNZ7\Desktop\codes\WorkRep\trunk\yolov4\yolov4-csp_pretrained.pytorch"))
# #%% Test without dataloader
# img = Image.open(r"E:\Datasets\Dense\cam_stereo_left_lut\2018-02-03_20-48-35_00500.png")
# img = torch.unsqueeze(tensor(img.resize((416,416))), 0)
# model = Darknet(nclasses=nc, anchors=anchors_g.astype(float))
# model.load_state_dict(torch.load(r"C:\Users\TK6YNZ7\Desktop\codes\WorkRep\trunk\yolov4\yolov4-csp_pretrained.pytorch"))
# model.eval()
# pred = model(img)
# boxes = non_max_suppression(pred[0].detach())[0]

# plt.rcParams['figure.figsize'] = (20,20)
# fig,ax = plt.subplots(1)
# #ax.imshow(image(torch.squeeze(img)))
# ax.imshow(image(img[0]))
# for i,box in enumerate(boxes):
#     xmin = box[0]
#     ymin = box[1]
#     width = (box[2]-box[0])
#     height = (box[3]-box[1])
#     rect = patches.Rectangle((xmin,ymin),width,height,linewidth=2,edgecolor='r',facecolor='none')
#     ax.add_patch(rect)
#     #ax.text(xmin, ymin, names[int(box[-1])], fontsize = 20)

# #%% Test with Dataloader
# train_set = Dataset(imroot,lroot,augment=False)
# train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=1,collate_fn=Dataset.collate_fn,shuffle=True)
# imgs,labels,shapes,paths = next(iter(train_loader))
# imgs = imgs/255.0
# model = Darknet(nclasses=nc, anchors=anchors_g)
# model.load_state_dict(torch.load(r"C:\Users\TK6YNZ7\Desktop\codes\WorkRep\trunk\yolov4\yolov4-csp_pretrained.pytorch"))
# model.eval()
# inf_out, train_out = model(imgs.float())
# output = non_max_suppression(inf_out)[0]
# #img = imgs[0].permute(1, 2, 0)
# #Visualise data loader
# # load a batch
# labels = scale_coords(img.shape, output[:, :4], shapes[0][0], shapes[0][1])
# # visualize batch
# original_img = Image.open(os.path.join(imroot,paths[0]))
# plt.rcParams['figure.figsize'] = (20,20)
# fig,ax = plt.subplots(1)
# #ax.imshow(image(torch.squeeze(img)))
# ax.imshow(original_img)
# for box,conf in zip(labels,output):
#     xmin = box[0]
#     ymin = box[1]
#     width = (box[2]-box[0])
#     height = (box[3]-box[1])
#     rect = patches.Rectangle((xmin,ymin),width,height,linewidth=2,edgecolor='r',facecolor='none')
#     ax.add_patch(rect)
#     ax.text(xmin, ymin, (names[int(conf[-1])],int(conf[-2]*100)), fontsize = 12)
# #%% visualise dataloader
# train_set = Dataset(imroot,lroot,augment=True)
# train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=1,collate_fn=Dataset.collate_fn,shuffle = True)
# imgs,labels,shapes = next(iter(train_loader))
# img = imgs[0].permute(1, 2, 0)
# w = img.shape[1]
# h = img.shape[0]
# plt.rcParams['figure.figsize'] = (20,20)
# fig,ax = plt.subplots(1)
# ax.imshow(img)
# for i,box in enumerate(labels):
#     xmin = (box[2]-box[4]/2)*w
#     ymin = (box[3]-box[5]/2)*h
#     width = box[4]*w
#     height =box[5]*h
#     rect = patches.Rectangle((xmin,ymin),width,height,linewidth=2,edgecolor='r',facecolor='none')
#     ax.add_patch(rect)
