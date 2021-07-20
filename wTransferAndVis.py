import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.cuda import amp
from torchviz import make_dot
import torch.optim as optim
import torch.nn as nn
from torch.nn.functional import sigmoid, softmax
from torch.distributions import Categorical
from torchvision import models, transforms
import os

from Fusion.models.models import *
from Fusion.utils.datasets import *
from FLIR_PP.arg_parser import DATASET_PP_PATH


dict_= {
    'device':'cuda', #Intialise device as cpu. Later check if cuda is avaialbel and change to cuda
    'device_num': '0',
    'anchors_g': [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]],
    'nclasses': 3, #Number of classes
    'names' : ['person', 'bicycle', 'car'],
    'img_size': 320, #Input image size. Must be a multiple of 32
    'img_format': '.jpg',
    'batch_size': 1, #train batch size
    'mode': 'rgb',
    'weight_path': './runs/train/exp_RGB320_300noMSnoMos/weights/best_ap50.pt',
    'task': 'test', # change to test only for the final test,
    'test_path' : DATASET_PP_PATH + '/Train_Test_Split/dev/',
    }
# dict_ = {
# 'device':'cuda', #Intialise device as cpu. Later check if cuda is avaialbel and change to cuda
# 'device_num': '0',
# 'anchors_g': [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]],
# 'nclasses': 3, #Number of classes
# 'names' : ['person', 'bicycle', 'car'],
# 'img_size': 320, #Input image size. Must be a multiple of 32
# 'strides': [8,16,32], #strides of p3,p4,p5
# 'epochs': 1, #number of epochs
# 'batch_size': 1, #train batch size
# 'test_size': 1, #test batch size
# 'use_adam': False, #Bool to use Adam optimiser
# 'use_ema': True, #Exponential moving average control
# 'multi_scale': True, #Bool to do multi-scale training
# 'gr' : 1.0, # giou loss ratio (obj_loss = 1.0 or giou)
# 'nms_conf_t':0.001, #0.2 Confidence training threshold
# 'project': './runs/train',
# 'comment': '_Series',
# 'test_all': True, #Run test after end of each epoch
# 'save_all': True, #Save checkpoints after every epoch
# 'plot': True,
# 'log_imgs': 16,
# 'resume': False,
# 'global_rank': -1,
# 'world_size': 1,
# 'local_rank': -1,
# 'sync_bn': False,
# 'workers': 8,
# 'cache_images': True,
# 'rect': False,
# 'image_weights': True,
# 'img_format': '.jpeg',
# 'train_aug' : True,
# 'evolve': False,
# 'task': 'val',
# 'train_path': DATASET_PP_PATH + '/Train_Test_Split/train/',
# 'val_path': DATASET_PP_PATH + '/Train_Test_Split/dev/',
# }


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

    torch.save(weight_dict, './RGBT_pre.pt')


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


def gui_vis():
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

def load_weights(model, dict_):
    try:
        ckpt = torch.load(dict_['weight_path']) # load checkpoint
        if ckpt['epoch'] != -1: print('Saved @ epoch: ', ckpt['epoch'])
        ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(ckpt['model'], strict=False)
    except:
        raise ValueError('The Weight does not exist!')
    
    return model

def conv_vis(model, dict_):
    try:
        ckpt = torch.load(dict_['weight_path']) # load checkpoint
        if ckpt['epoch'] != -1: print('Saved @ epoch: ', ckpt['epoch'])
        ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(ckpt['model'], strict=False)
    except:
        raise ValueError('The Weight does not exist!')

    model_weights = [] # we will save the conv layer weights in this list
    conv_layers = [] # we will save the 49 conv layers in this list
    num_of_layers = 0


    # for i, key in enumerate(ckpt['model']):
    #     if 'weight' in key and 'batchnorm' not in key: # 'conv' in key and 
    #         model_weights.append(ckpt['model'][key])
    #         conv_layers.append(key)
    #         num_of_layers += 1
    #         # print(key)
    # print(f"Total convolutional layers: {counter}")

    # print(model)
    model_children=list(model.children())

    # get all the conv. layers in sub-branches
    for child in model_children:
        child2 = child.children()
        for layer in child2:
            if type(layer)==nn.Conv2d:
                    num_of_layers+=1
                    conv_layers.append(layer)
            else:
                child3 = layer.children()
                for layer in child3:
                    if type(layer)==nn.Conv2d:
                        num_of_layers+=1
                        conv_layers.append(layer)
                    else:
                        child4 = layer.children()
                        for layer in child4:
                            if type(layer)==nn.Conv2d:
                                num_of_layers+=1
                                conv_layers.append(layer)
                            else:
                                child5 = layer.children()
                                for layer in child5:
                                    if type(layer)==nn.Conv2d:
                                        num_of_layers+=1
                                        conv_layers.append(layer)
                                    else:
                                        child6 = layer.children()
                                        for layer in child6:
                                            if type(layer)==nn.Conv2d:
                                                num_of_layers+=1
                                                conv_layers.append(layer)
                                            else:
                                                child7 = layer.children()
                                                for layer in child7:
                                                    if type(layer)==nn.Conv2d:
                                                        num_of_layers+=1
                                                        conv_layers.append(layer)
                                                    else:
                                                        child7 = layer.children()
                                                        for layer in child7:
                                                            if type(layer)==nn.Conv2d:
                                                                num_of_layers+=1
                                                                conv_layers.append(layer)

    i = 0
    for element in ckpt['model']:
    # # for i in range (len(ckpt['model'])):
        if 'conv.weight' in element or 'head.final3.weight' in element\
                                    or 'head.final4.weight' in element\
                                    or 'head.final5.weight' in element:
            print(conv_layers[i], '\n', element, ' Conv-idx:{}'.format(i), '\n')
            i +=1


    #         # print(element)
    #         model_weights.append(ckpt['model'][element])

    # print(num_of_layers)

    # # Kernel Vis.
    # plt.figure(figsize=(20, 17))
    # # print(model_weights[0].shape)
    # for i, filter in enumerate(model_weights[0]):
    #     plt.subplot(8, 8, i+1) # (4, 4) because in conv0 we have 3x3 filters and total of 32 (see printed shapes)
    #     # print(filter.shape)
    #     plt.imshow(filter[:, :, :].detach().cpu()) # , cmap='gray'
    #     plt.axis('off')
    #     # plt.savefig('../outputs/filter.png')
    # plt.show()

    # read and visualize an image
    img = cv2.imread(os.path.join(DATASET_PP_PATH, 'samples/FLIR_00145.jpg')) # 150 158 163 164 165
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(img)
    # plt.show()

    # define the transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
    ])
    img = np.array(img)
    # apply the transforms
    img = transform(img)
    # print(img.size())
    # unsqueeze to add a batch dimension
    img = img.unsqueeze(0)
    # print(img.size())

    # pass the image through all the layers
    results = [conv_layers[0](img)]
    for i in range(1, len(conv_layers)):
        print('\nW_in: ', conv_layers[i].in_channels,
              '\nW_out: ', conv_layers[i].out_channels,
              '\nW_in_next:', conv_layers[i+1].in_channels,
              '\nConvL: ', results[-1].shape)
        # pass the result from the last layer to the next layer
        try:
            results.append(conv_layers[i](results[-1]))
        except:
            # swap the current layer with the next layer
            temp = conv_layers[i]
            conv_layers[i] = conv_layers[i+1]
            conv_layers[i+1] = temp
            results.append(conv_layers[i](results[-1]))
    # make a copy of the `results`
    outputs = results

    # visualize 64 features from each layer 
    # (although there are more feature maps in the upper layers)
    for num_layer in range(len(outputs)):
        plt.figure(figsize=(30, 30))
        layer_viz = outputs[num_layer][0, :, :, :]
        layer_viz = layer_viz.data
        print(layer_viz.size())
        for i, filter in enumerate(layer_viz):
            if i == 64: # we will visualize only 8x8 blocks from each layer
                break
            plt.subplot(8, 8, i + 1)
            plt.imshow(filter, cmap='gray') # 
            plt.axis("off")
        print(f"Saving layer {num_layer} feature maps...")
        plt.savefig(f"./layer_{num_layer}.png")
        # plt.show()
        plt.close()

if '__main__' == __name__:

    #=================================================================================
    # Saving the state dict without weights

    # dict_['mode'] = 'fusion'
    # model = Fused_Darknets(dict_, (640, 640)).to('cuda') # create
    # torch.save(model.state_dict(), './RGBT_init.pt')


    # dict_['mode'] = 'ir'
    # model = Darknet(dict_, (640, 640)).to('cuda') # create
    # torch.save(model.state_dict(), './IR_init.pt')

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

    # fusion_state_dict = torch.load("./RGBT_init.pt")
    # ir_state_dict = torch.load("./runs/train/exp_IR320_300noMSnoMos/weights/best_val_loss.pt")['model']
    # rgb_state_dict = torch.load("./runs/train/exp_RGB320_300noMSnoMos/weights/best_val_loss.pt")['model']
    # # ir_state_dict = torch.load("./IR.pt")['model']
    # # rgb_state_dict = torch.load("./yolo_pre_3c.pt")['model']
    # BLs_W_to_RGBT(rgb_state_dict, ir_state_dict, fusion_state_dict)

    #=================================================================================
    # Weight Transfer for IR

    # ir_init_state_dict = torch.load("./IR_init.pt")
    # # ir_state_dict = torch.load("./runs/train/exp_IR_BL_640_100ms-from44RGB_2/weights/best_ap50.pt")['model']
    # ir_state_dict = torch.load("./yolo_pre_3c.pt")['model']
    # ir_weight_transfer(ir_state_dict, ir_init_state_dict)

    #=================================================================================
    # Conv. Visualization

    # model = Darknet(dict_, img_size=(dict_['img_size'], dict_['img_size']))
    # # print(model)
    # conv_vis(model, dict_)

    #=================================================================================
    # # GFLOPS
    # from fvcore.nn import FlopCountAnalysis
    # from fvcore.nn import flop_count_table
    # from fvcore.nn import flop_count_str
    # from Fusion.utils.torch_utils import select_device

    # device = select_device(device=dict_['device_num'], batch_size=dict_['batch_size'])
    # model = Darknet(dict_, img_size=(dict_['img_size'], dict_['img_size'])).to(device)
    
    # img = cv2.imread(os.path.join(DATASET_PP_PATH, 'samples/FLIR_00145.jpg'))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize((320, 320)),
    #     transforms.ToTensor(),
    # ])
    # img = np.array(img)
    # img = transform(img)
    # # unsqueeze to add a batch dimension
    # img = img.unsqueeze(0)
    # img = img.to(device)

    # flops = FlopCountAnalysis(model, img)
    # print(flop_count_table(flops))
    # # print(flops.by_module_and_operator()) # flops.by_module(), flops.by_operator()

    # #-------------------------------------------------------------------------------
    # # FPS

    # model = load_weights(model, dict_)
    # model.eval()

    # # Half
    # half = device.type != 'cpu'  # half precision only supported on CUDA
    # if half:
    #     model.half()

    # # Run Once
    # imgsz = img.shape
    # img_test = torch.zeros((1, 3, imgsz[-1], imgsz[-1]), device=device)  # init img
    # _ = model(img_test.half() if half else img_test) if device.type != 'cpu' else None  # run once

    # img = img.to(device, non_blocking=True)
    # img = img.half() if half else img.float()  # uint8 to fp16/32
    # img /= 255.0  # 0 - 255 to 0.0 - 1.0
    # if len(img.shape) == 3: #  ir mode
    #     img = torch.unsqueeze(img, axis=1)

    # RUN_NUM = 30
    # WARM_UP_RUNS = 5

    # with torch.no_grad():
    #     for i in range(RUN_NUM):
    #         if i == WARM_UP_RUNS:
    #             start_time = time.time()
    #         pred = model(img.half() if half else img) if device.type != 'cpu' else None

    # end_time = time.time()
    # total_forward = end_time - start_time
    # print('Total forward time is %4.2f seconds' % total_forward)
    # actual_num_runs = RUN_NUM - WARM_UP_RUNS
    # latency = total_forward / actual_num_runs
    # fps = 1/latency
    # # fps = (cfg.CONFIG.DATA.CLIP_LEN * cfg.CONFIG.DATA.FRAME_RATE) * actual_num_runs / total_forward

    # print("FPS: ", fps, "; Latency: ", latency)

    #=================================================================================
    # # Entropy
    # p = np.array([0.1, 0.2, 0.4, 0.3])
    # p = np.array([0.1, 0.1, 0.1, 1])
    # logp = np.log(p)
    # entropy1 = np.sum(-p*logp)
    # print(entropy1)

    # tensor = torch.Tensor([[[[0.1, 0.2], [0.1, 0.2]], [[0.1, 0.2], [0.1, 0.2]], [[0.1, 0.2], [0.1, 0.2]], [[0.4, 0.3], [0.1, 0.2]]],
    #                          [[[0.1, 0.2], [0.1, 0.2]], [[0.4, 0.3], [0.1, 0.2]], [[0.4, 0.3], [0.1, 0.2]], [[0.4, 0.3], [0.1, 0.2]]]])

    tensor = torch.Tensor([[[5,4], [4,5], [5,4], [4,5]]])
    tensor = tensor.reshape((1,8)).softmax(dim=1).reshape((1,4,2))
    # tensor *= 10
    tensor_p = softmax(tensor, dim=2)#.permute(0, 2, 3, 1)
    print(tensor_p)
    entropy2 = Categorical(probs = tensor_p).entropy().unsqueeze(dim=1)
    print(entropy2)
    tensor *= entropy2
    print(tensor)

    # def entropy(self, x):
    #     prob_x = softmax(x, dim=1).permute(0, 2, 3, 1)
    #     entropy = Categorical(probs = prob_x).entropy().unsqueeze(dim=1)
    #     return x*entropy