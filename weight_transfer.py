import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.cuda import amp
from torchviz import make_dot
import torch.optim as optim

from Fusion.models.models import *
from Fusion.utils.datasets import *
from FLIR_PP.arg_parser import DATASET_PP_PATH


# Weight transfer from baselines to Fusion Network
def BLs_W_to_RGBT(rgb_state_dict, ir_state_dict, fusion_state_dict):
    rgb_keys = list(rgb_state_dict)
    ir_keys = list(ir_state_dict)

    done_jobs = 0
    for i, key in enumerate(fusion_state_dict):
        if 'backbone' and 'rgb' in key:
            fusion_state_dict[key] = rgb_state_dict[rgb_keys[done_jobs]] # i from 0:408
            done_jobs += 1
        elif 'backbone' and 'ir' in key: #  and not ('f_x3_Conv2d' or 'f_x4_Conv2d' or 'f_x5_Conv2d')
            fusion_state_dict[key] = ir_state_dict[ir_keys[done_jobs-408]] # i from 408:815
            done_jobs += 1
        elif 'backbone' not in key:
            fusion_state_dict[key] = ir_state_dict[ir_keys[done_jobs-408]]
            done_jobs += 1

    weight_dict = {'epoch':0,
                'best_fitness': 0,
                'best_fitness_p': 0,
                'best_fitness_r': 0,
                'best_fitness_ap50': 0,
                'best_fitness_ap': 0,
                'best_fitness_f': 0,
                'training_results': None,
                'model': fusion_state_dict,
                'optimizer': None,
                'wandb_id': None
                }

    torch.save(weight_dict, './RGBT.pt')


def ir_weight_transfer(ir_state_dict, ir_init_state_dict):
    ir_keys = list(ir_state_dict)

    for i, key in enumerate(ir_init_state_dict):
        ir_init_state_dict[key] = ir_state_dict[ir_keys[i]]

    weight_dict = {'epoch':0,
                'best_fitness': 0,
                'best_fitness_p': 0,
                'best_fitness_r': 0,
                'best_fitness_ap50': 0,
                'best_fitness_ap': 0,
                'best_fitness_f': 0,
                'training_results': None,
                'model': ir_init_state_dict,
                'optimizer': None,
                'wandb_id': None
                }

    torch.save(weight_dict, './IR.pt')


def visualizations():
    import cv2
    img1 = cv2.imread('/data/Sam/FLIR/FLIR_PP/Train_Test_Split/test_Night/FLIR_03718.jpeg', 0)
    img2 = cv2.imread('/data/Sam/FLIR/FLIR_PP/Train_Test_Split/test_Night/FLIR_03718.jpg')

    shape = img1.shape
    shape = img1.shape

    img1 = np.asarray(img1)[..., np.newaxis]
    # img2 = np.asarray(img2)
    img = np.concatenate((img2, img1), axis=-1)
    img = img[..., -1]
    cv2.imshow('asd', img)
    cv2.waitKey()
    print(img1.shape)
    print(img2.shape)
    print(img.shape)

    # ir = plt.imread('/data/Sam/FLIR/FLIR_PP/Train_Test_Split/dev/FLIR_00015.jpg')
    # plt.imshow(img)
    # plt.show()



if '__main__' == __name__:
    dict_ = {
    'device':'cuda', #Intialise device as cpu. Later check if cuda is avaialbel and change to cuda
    'device_num': '0',
    'anchors_g': [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]],
    'nclasses': 3, #Number of classes
    'names' : ['person', 'bicycle', 'car'],
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
    'project': './runs/train',
    'comment': '_Series',
    'test_all': True, #Run test after end of each epoch
    'save_all': True, #Save checkpoints after every epoch
    'plot': True,
    'log_imgs': 16,
    'resume': False,
    'global_rank': -1,
    'world_size': 1,
    'local_rank': -1,
    'sync_bn': False,
    'workers': 8,
    'cache_images': True,
    'rect': False,
    'image_weights': True,
    'img_format': '.jpeg',
    'train_aug' : True,
    'evolve': False,
    'task': 'val',
    'train_path': DATASET_PP_PATH + '/Train_Test_Split/train/',
    'val_path': DATASET_PP_PATH + '/Train_Test_Split/dev/',
    }

    #=================================================================================
    # Saving the state dict without weights

    dict_['mode'] = 'fusion'

    model = Fused_Darknets(dict_, (640, 640)).to('cuda') # create
    # model = Darknet(dict_, (640, 640)).to('cuda') # create
    torch.save(model.state_dict(), './init.pt')

    # loading a sample weight to check the structure of the dict
    # ckpt = torch.load('/home/efs-gx/RGBT/runs/train/exp_RGB320_default/weights/best.pt', map_location='cuda')  # load checkpoint
    # ckpt = torch.load('/home/efs-gx/RGBT/runs/train/exp_IR_BL_640_100ms-from44RGB_2/weights/best.pt', map_location='cuda')  # load checkpoint
    # ckpt = torch.load('RGBT.pt', map_location='cuda')  # load checkpoint

    #=================================================================================
    # Plotting the graph

    # model.eval()
    # rgb = torch.zeros((1, 3, 640, 640)).to('cuda')
    # ir = torch.zeros((1, 3, 640, 640)).to('cuda')
    # inf_out, train_out = model(rgb, ir)

    # dot = make_dot(inf_out)
    # dot.format = 'jpeg'
    # dot.render('FD2')

    #=================================================================================
    # Weight Transfer for Fusion

    fusion_state_dict = torch.load("./init.pt")
    # ir_state_dict = torch.load("./runs/train/exp_IR_BL_640_100ms-from44RGB_2/weights/best_ap50.pt")['model']
    # rgb_state_dict = torch.load("./runs/train/exp_RGB_from_Scratch_4/weights/last.pt")['model']
    ir_state_dict = torch.load("./IR.pt")['model']
    rgb_state_dict = torch.load("./yolo_pre_3c.pt")['model']
    BLs_W_to_RGBT(rgb_state_dict, ir_state_dict, fusion_state_dict)

    #=================================================================================
    # Weight Transfer for IR

    # ir_init_state_dict = torch.load("./init.pt")
    # # ir_state_dict = torch.load("./runs/train/exp_IR_BL_640_100ms-from44RGB_2/weights/best_ap50.pt")['model']
    # ir_state_dict = torch.load("./yolo_pre_3c.pt")['model']
    # ir_weight_transfer(ir_state_dict, ir_init_state_dict)