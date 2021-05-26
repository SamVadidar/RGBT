import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from Fusion.models.experimental import attempt_load
from Fusion.utils.datasets import create_dataloader
from Fusion.utils.general import (
    coco80_to_coco91_class, check_file, check_img_size, compute_loss, non_max_suppression,
    scale_coords, xyxy2xywh, clip_coords, plot_images, xywh2xyxy, box_iou, output_to_target, ap_per_class)
from Fusion.utils.torch_utils import select_device, time_synchronized
from Fusion.models.models import *
from FLIR_PP.arg_parser import DATASET_PP_PATH


# def load_classes(path):
#     with open(path, 'r') as f:
#         names = f.read().split('\n')
#     return list(filter(None, names))  # filter removes empty strings (such as last line)


def test(dict,
         hyp,
         model=None,
         augment=False,
         verbose=False,
         dataloader=None):

    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        merge = dict['nms_merge']
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        device = select_device(device=dict['device_num'], batch_size=dict['batch_size'])
        merge, save_txt = dict['nms_merge'], dict['save_txt']  # use Merge NMS, save *.txt labels
        if save_txt:
            out = Path('inference/output')
            if os.path.exists(out):
                shutil.rmtree(out)  # delete output folder
            os.makedirs(out)  # make new output folder

        # Remove previous
        for f in glob.glob(str(Path(dict['save_dir']) / 'test_batch*.jpg')):
            os.remove(f)


        img_size = dict['img_size']
        model = Darknet(dict, (img_size, img_size)).to(device)

        # load model
        try:
            ckpt = torch.load(dict['weight_path']) # load checkpoint
            if ckpt['epoch'] != -1: print('Saved @ epoch: ', ckpt['epoch'])
            ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(ckpt['model'], strict=False)
        except:
            raise ValueError('The Weight does not exist!')

        imgsz = check_img_size(dict['img_size'], s=32)  # check img_size

    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    model = model.eval()
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        # _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        path = dict['test_path'] if dict['validation_mode'] == 'test' else dict['val_path']  # path to val/test images
        pad = 0 if dict['plot_all'] else 0.5 #  org pad=0.5
        dataloader = create_dataloader(path, imgsz, dict['batch_size'], 32,
                                       hyp=None, augment=False, cache=False, pad=pad, rect=True)[0] # grid_size=32

    seen = 0

    names = dict['names']
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Disable gradients
        with torch.no_grad():
            # Run model
            t = time_synchronized()
            inf_out, train_out = model(img, augment=augment)  # inference and training outputs
            t0 += time_synchronized() - t

            # Compute loss
            if training:  # if model has loss hyperparameters
                loss += compute_loss([x.float() for x in train_out], targets, hyp, dict)[1][:3]  # GIoU, obj, cls
                # loss += compute_loss([x.float() for x in train_out], targets, hyp, dict)[1][:3]  # GIoU, obj, cls

            # Run NMS
            t = time_synchronized()
            output = non_max_suppression(inf_out, conf_thres=dict['nms_conf_t'], iou_thres=hyp['iou_t'], merge=merge)
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
                continue

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

        # Plot images
        if batch_i < 1:
            if not os.path.isdir(dict['save_dir']): os.mkdir(dict['save_dir'])
            f = Path(dict['save_dir']) / ('test_batch%g_gt.jpg' % batch_i)  # filename
            plot_images(img, targets, paths, './'+str(f), names)  # ground truth
            f = Path(dict['save_dir']) / ('test_batch%g_pred.jpg' % batch_i)
            plot_images(img, output_to_target(output, width, height), paths, str(f), names)  # predictions

        if dict['plot_all']:
            for si, pred in enumerate(output):
                labels = targets[targets[:, 0] == si, 1:]
                # imgs = img[si].permute(1, 2, 0).cpu()
                # imgs_255 = torch.mul(imgs, 255)
                original_img = Image.open(os.path.join(paths[si]))
                # width,height,_ = original_img.shape  # Pytorch Tensor
                width,height = original_img.size # PIL Image
                plt.rcParams['figure.figsize'] = (20,20)
                fig,ax = plt.subplots(1)
                ax.imshow(original_img)
                if pred is not None:
                    boxes = pred[:,:4]
                    boxes[:, :4] = scale_coords(img[si].shape[1:], boxes[:, :4], shapes[si][0], shapes[si][1])  # to original
                    for i, (box,label) in enumerate(zip(boxes.cpu(),labels.cpu())):
                        xmin = box[0]
                        ymin = box[1]
                        w = (box[2]-box[0])
                        h = (box[3]-box[1])
                        rect = patches.Rectangle((xmin,ymin),w,h,linewidth=2,edgecolor='r',facecolor='none')
                        ax.add_patch(rect)
                        # ax.text(xmin, ymin, '%s %s'%(dict['names'][int(box[-1])],int(box[-2]*100)/100), fontsize = 12)
                        x = (label[1]-label[3]/2)*width
                        y = (label[2]-label[4]/2)*height
                        wid = label[3]*width
                        hei = label[4]*height
                        rect1 = patches.Rectangle((x,y),wid,hei,linewidth=2,edgecolor='g',facecolor='none')
                        ax.add_patch(rect1)
                    plt.show()
                    plt.close()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=dict['nclasses'])  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if verbose and dict['n_classes'] > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (dict['img_size'], dict['img_size'], dict['test_size'])  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # # org below
    # # Save JSON
    # if save_json and len(jdict):
    #     f = 'detections_val2017_%s_results.json' % \
    #         (weights.split(os.sep)[-1].replace('.pt', '') if isinstance(weights, str) else '')  # filename
    #     print('\nCOCO mAP with pycocotools... saving %s...' % f)
    #     with open(f, 'w') as file:
    #         json.dump(jdict, file)

    # Return results
    model.float()  # for training
    maps = np.zeros(dict['nclasses']) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    dict_ = {
        'device':'cuda', #Intialise device as cpu. Later check if cuda is avaialbel and change to cuda
        'device_num': '0',

        # Kmeans on COCO
        'anchors_g': [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]],

        # # Kmeans on FLIR
        # 'anchors_g': [[7, 16], [16, 15], [12, 30], [28, 24], [18, 57], [49, 39], [43, 99], [113, 74], [163, 168]],

        'nclasses': 3, #Number of classes
        'names' : ['person', 'bicycle', 'car'],
        'gs': 32, #Image size multiples
        'img_size': 320, #Input image size. Must be a multiple of 32
        'batch_size': 16, #train batch size
        'test_size': 16, #test batch size
        'gr' : 1.0, # giou loss ratio (obj_loss = 1.0 or giou)

        # Data loader
        'rect': True,

        # test
        'nms_conf_t':0.001, #Confidence test threshold
        'nms_merge': True,

        #logs
        'save_txt': True,
        'plot_all': False,
        'save_dir': './save_dir/',

        # PATH
        'weight_path': './miniRuns/exp3_FinalBL/weights/best_.pt',
        'train_path': DATASET_PP_PATH + '/Train_Test_Split/train/',
        'val_path': DATASET_PP_PATH + '/Train_Test_Split/dev/',
        'test_path' : DATASET_PP_PATH + '/Train_Test_Split/test/',

        'validation_mode': 'test', # change to test for the final test
     }

    hyp = {
        # test
        'iou_t': 0.6,  # IoU test threshold
    }

    test(dict_, hyp, augment=False)
