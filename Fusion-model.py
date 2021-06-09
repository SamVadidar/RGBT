from Fusion.models.models import *
from Fusion.utils.datasets import *
from FLIR_PP.arg_parser import DATASET_PP_PATH

from torch.cuda import amp
from torchviz import make_dot
import torch.optim as optim


dict_ = {
    'device':'cuda', #Intialise device as cpu. Later check if cuda is avaialbel and change to cuda
    'device_num': '0',

    # Kmeans on COCO
    'anchors_g': [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]],

    # # Kmeans on FLIR
    # 'anchors_g': [[7, 16], [16, 15], [12, 30], [28, 24], [18, 57], [49, 39], [43, 99], [113, 74], [163, 168]],

    'nclasses': 3, #Number of classes
    'names' : ['person', 'bicycle', 'car'],
    # 'gs': 32, #Image size multiples
    'img_size': 320, #Input image size. Must be a multiple of 32
    'strides': [8,16,32], #strides of p3,p4,p5
    'epochs': 1, #number of epochs
    'batch_size': 1, #train batch size
    'test_size': 1, #test batch size
    'use_adam': False, #Bool to use Adam optimiser
    'use_ema': True, #Exponential moving average control
    'multi_scale': True, #Bool to do multi-scale training
    'gr' : 1.0, # giou loss ratio (obj_loss = 1.0 or giou)
    'nms_conf_t':0.001, #0.2 Confidence training threshold

    #logs
    'project': './runs/train',
    # 'logdir': './miniRuns',
    'comment': '_Series',
    'test_all': True, #Run test after end of each epoch
    'save_all': True, #Save checkpoints after every epoch
    'plot': True,
    'log_imgs': 16,
    'resume': False,

    # DP
    'global_rank': -1,
    'world_size': 1,
    'local_rank': -1,
    'sync_bn': False,

    # Data loader
    'workers': 8,
    'cache_images': True,
    'rect': False,
    'image_weights': True,
    'img_format': '.jpg',
    'train_aug' : True,

    # Hyp. Para.
    'evolve': False,

    # TODO: Image Format, , Comment, Weight_path, Img size, Aug., train/val set

    # PATH
    # 'weight_path': './miniRuns/exp8_mini320IR/weights/last.pt',
    # 'weight_path': './yolo_pre_3c.pt',
    # 'weight_path': './runs/train/exp_640RGB100ms4/weights/last.pt',
    'task': 'val',

    # large
    'train_path': DATASET_PP_PATH + '/Train_Test_Split/train/',
    'val_path': DATASET_PP_PATH + '/Train_Test_Split/dev/',

    # mini
    # 'train_path': DATASET_PP_PATH + '/mini_Train_Test_Split/train/',
    # 'val_path': DATASET_PP_PATH + '/mini_Train_Test_Split/dev/',
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
    'obj': 1.0,  # obj loss gain (scale with pixels)
    'obj_pw': 1.0,  # obj BCELoss positive_weight
    'iou_t': 0.2,  # IoU training threshold
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
    'translate': 0.1,  # image translation (+/- fraction)
    'scale': 0.9,  # image scale (+/- gain)
    'shear': 0.0,  # image shear (+/- deg)
    'perspective': 0.0,  # image perspective (+/- fraction), range 0-0.001
    'flipud': 0.0,  # image flip up-down (probability)
    'fliplr': 0.5,  # image flip left-right (probability)
    'mosaic': 1.0,
    'mixup': 0.0, #mix up probability
}

model = Fused_Darknets(dict_, (640, 640)).to('cuda') # create
torch.save(model.state_dict(), './init.pt')

# model.eval()
# rgb = torch.zeros((1, 3, 640, 640)).to('cuda')
# ir = torch.zeros((1, 3, 640, 640)).to('cuda')
# inf_out, train_out = model(rgb, ir)

# dot = make_dot(inf_out)
# dot.format = 'jpeg'
# dot.render('FD2')

# dataloader, dataset = create_dataloader(dict_['train_path'], dict_['img_size'], 1, 64, hyp=hyp, augment=False,
#                                         cache=dict_['cache_images'], rect=dict_['rect'], rank=dict_['local_rank'],
#                                         world_size=dict_['world_size'], workers=dict_['workers'], img_format=dict_['img_format'])

# pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
# for k, v in dict(model.named_parameters()).items():
#     if '.bias' in k:
#         pg2.append(v)  # biases
#     elif 'conv.weight' in k: # or '1.weight'in k:
#         pg1.append(v)  # apply weight_decay
#     elif k in ['head.final3.weight','head.final4.weight','head.final5.weight']:
#         pg1.append(v)
#     else:
#         pg0.append(v)  # all else
# optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

# nb = len(dataloader)  # number of batches
# model.train()
# pbar = enumerate(dataloader)
# if dict_['local_rank'] in [-1, 0]:
#     pbar = tqdm(pbar, total=nb)  # progress bar
# if dict_['local_rank'] in [-1, 0]:
#     pbar = tqdm(pbar, total=nb)  # progress bar
# optimizer.zero_grad()
# for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
#     ni = i + nb * 10  # number integrated batches (since train start)
#     imgs = imgs.to('cuda', non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

    # Forward
# with amp.autocast(enabled=True):
#     pred = model(torch.zeros((1, 3, 640, 640)).to('cuda'))
#     dot = make_dot(pred[0])
#     dot.format = 'jpg'
#     dot.render('FD')