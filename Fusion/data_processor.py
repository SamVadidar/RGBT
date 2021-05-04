# -*- coding: utf-8 -*-
"""
Adapted From: WongKinYiu and Gokulesh Danapal
https://github.com/WongKinYiu/ScaledYOLOv4
https://github.com/gokulesh-danapal
"""
import os
import random
import torch
import torchvision as tv
from Fusion.utils import xyxy2xywh, create_mosaic, letterbox, load_image, load_label


class Dataset(object):
  def __init__(self,hyp,imroot,lroot,augment = True):
    self.augment = augment
    self.imroot = imroot
    #self.rroot = rroot
    self.lroot = lroot
    self.inputs = list(os.listdir(imroot))
    self.fliplr = tv.transforms.RandomHorizontalFlip(p=1)
    self.flipud = tv.transforms.RandomVerticalFlip(p=1)
    self.hyp = hyp
  def __getitem__(self,index):
    # load images
    shapes = None
    if self.augment:
        img, labels = create_mosaic(self.imroot, self.lroot,index, self.hyp)
         #MixUp https://arxiv.org/pdf/1710.09412.pdf
        if random.random() < self.hyp['mixup']:
            img2, labels2 = create_mosaic(self.imroot, self.lroot,random.randint(0, len(labels) - 1),self.hyp)
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
        img, (h0, w0), (h, w) = load_image(self.imroot,index,self.hyp)
        # Letterbox
        img, ratio, pad = letterbox(img, self.hyp['img_size'], auto=False, scaleup=self.augment)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
        # Load labels
        x = load_label(self.imroot,self.lroot, index)
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
        if random.random() < self.hyp['flipud']:
            img = np.flipud(img)
            labels[:, 2] = 1 - labels[:, 2]
        # flip left-right
        if random.random() < self.hyp['fliplr']:
          img = np.fliplr(img)
          labels[:, 1] = 1 - labels[:, 1]
    labels_out = torch.zeros((len(labels), 6))
    if nL:
        labels_out[:, 1:] = torch.from_numpy(labels)
    # Convert
    img = img.copy()
    img = img.transpose(2, 0, 1)  # BGR to RGB, to 3x416x416 [:, :, ::-1]
    #img = np.ascontiguousarray(img)
    return torch.from_numpy(img), labels_out, shapes, os.path.join(self.imroot,self.inputs[index])

  @staticmethod
  def collate_fn(batch):
    img, label,shapes,paths  = zip(*batch)  # transposed
    for i, l in enumerate(label):
        l[:, 0] = i  # add target image index for build_targets()
    return torch.stack(img, 0), torch.cat(label, 0), shapes, paths

  def __len__(self):
    return len(self.inputs)