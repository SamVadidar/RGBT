# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:53:12 2021

@author: Gokulesh Danapal (GX6)

Confidentiality: Internal
"""
from yolo_backend import Dataset, Darknet, train, test
from torch.utils.tensorboard import SummaryWriter

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']

hyp = { 'device':'cuda', #Intialise device as cpu. Later check if cuda is avaialbel and change to cuda    
        'lr0': 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
        'momentum': 0.937,  # SGD momentum/Adam beta1
        'weight_decay': 0.0005,  # optimizer weight decay
        'anchors_g': [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]],
        'nclasses': 80, #Number of classes
        'img_size': 640, #Input image size. Must be a multiple of 32
        'strides': [8,16,32], #strides of p3,p4,p5
        'epochs': 10, #number of epochs
        'batch_size': 16, #train batch size
        'test_size': 1, #test batch size
        'use_adam': False, #Bool to use Adam optimiser
        'multi_scale': False, #Bool to do multi-scale training
        'test_all': False, #Run test after end of each epoch
        'save_all': True, #Save checkpoints after every epoch
        
        'giou': 0.05,  # GIoU loss gain
        'cls': 0.5,  # cls loss gain
        'cls_pw': 1.0,  # cls BCELoss positive_weight
        'obj': 1.0,  # obj loss gain (scale with pixels)
        'obj_pw': 1.0,  # obj BCELoss positive_weight
        'gr' : 1.0, # giou loss ratio (obj_loss = 1.0 or giou)
        'iou_t': 0.6,  # IoU training threshold
        'conf_t':0.5, # Confidence training threshold
        'anchor_t': 4.0,  # anchor-multiple threshold
        
        'fl_gamma': 0.0,  # focal loss gamma (efficientDet default gamma=1.5)
        'flipud': 0.0,  # image flip up-down (probability)
        'fliplr': 0.0,  # image flip left-right (probability)
        'mixup': 0.0 #mix up probability
     }


weight_path = './yolo_pre.pt'
imroot = '/home/ub145/Documents/Dataset/FLIR/FLIR/FLIR_PP/val/RGB_cropped/'
lroot = '/home/ub145/Documents/Dataset/FLIR/FLIR/FLIR_PP/val/yolo_format_labels'
# imroot = './asd/images'
# lroot = './asd/labels'
logdir = './runs'

train_set = Dataset(hyp,imroot,lroot,augment=True)
test_set = Dataset(hyp,imroot, lroot, augment= False)
tb_writer = SummaryWriter(log_dir = logdir)


#results = train(hyp,tb_writer, train_set, weight_path, test_set)

results = test(test_set,names,hyp,weight_path, plot_all=False)
