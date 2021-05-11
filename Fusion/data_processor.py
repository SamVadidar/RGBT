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
import numpy as np
import cv2
from matplotlib import pyplot as plt
from Fusion.utils import xyxy2xywh, create_mosaic, letterbox, load_image, load_label, random_perspective, augment_hsv


class Dataset(object):
    def __init__(self,hyp,imroot,lroot,augment = True, mosaic = True):
        self.augment = augment
        self.imroot = imroot
        self.lroot = lroot
        self.inputs = list(os.listdir(imroot))
        self.hyp = hyp
        self.mosaic = mosaic
    def __getitem__(self,index):
        # load images
        if self.mosaic:
            img, labels = create_mosaic(self.imroot, self.lroot, index, self.inputs, self.hyp)
            shapes = None
            #MixUp https://arxiv.org/pdf/1710.09412.pdf
            if random.random() < self.hyp['mixup']:
                img2, labels2 = create_mosaic(self.imroot, self.lroot,random.randint(0, len(labels) - 1), self.inputs, self.hyp)
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)

        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(os.path.join(self.imroot, self.inputs[index]),self.hyp['img_size'])
            # Letterbox
            img, ratio, pad = letterbox(img, self.hyp['img_size'], auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
            # Load labels
            x = load_label(os.path.join(self.lroot, self.inputs[index].replace('.jpg','.txt')))
            labels = x.copy()
            nL = len(x)
            if nL:
                labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]
                
        if self.augment:
                # Augment imagespace
                if not self.mosaic:
                    img, labels = random_perspective(img, labels,
                                                        degrees=self.hyp['degrees'],
                                                        translate=self.hyp['translate'],
                                                        scale=self.hyp['scale'],
                                                        shear=self.hyp['shear'],
                                                        perspective=self.hyp['perspective'])
                # Augment colorspace
                augment_hsv(img, hgain=self.hyp['hsv_h'], sgain=self.hyp['hsv_s'], vgain=self.hyp['hsv_v'])
                
        nL = len(labels)  # number of labels

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
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.show()
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