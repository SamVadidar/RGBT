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

from Fusion.utils.datasets import create_dataloader
from Fusion.utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, box_iou, \
    non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, clip_coords, set_logging, increment_path
from Fusion.utils.loss import compute_loss
from Fusion.utils.metrics import ap_per_class
from Fusion.utils.plots import plot_images, output_to_target
from Fusion.utils.torch_utils import select_device, time_synchronized

from Fusion.models.models import *
from FLIR_PP.arg_parser import DATASET_PP_PATH


def test(dict,
         hyp,
         model=None,
         augment=False,
         verbose=True,
         dataloader=None,
         save_txt = False,
         save_dir = Path(''), # for saving images
         log_imgs=0):

    # Initialize/load model and set device
    training = model is not None

    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(device=dict['device_num'], batch_size=dict['batch_size'])
        save_txt = dict['save_txt']

        # Directories
        save_dir = Path(increment_path((Path('./runs/test') / 'exp'), exist_ok=False, sep='_'))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

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

        imgsz = check_img_size(dict['img_size'], s=64)  # check img_size

    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    model.eval()
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs, wandb = min(log_imgs, 100), None  # ceil
    try:
        import wandb  # Weights & Biases

    except ImportError:
        log_imgs = 0

    # Dataloader
    if not training:
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        path = dict['test_path'] if dict['task'] == 'test' else dict['val_path']  # path to val/test images
        dataloader = create_dataloader(path, imgsz, dict['batch_size'], 64, hyp=hyp, augment=dict['aug'], pad=0.5, rect=True, img_format=dict['img_format'])[0] # grid_size=32

    seen = 0

    names = dict['names']
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
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
            output = non_max_suppression(inf_out, conf_thres=dict['nms_conf_t'], iou_thres=hyp['iou_t'])
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

            # Append to text file
            path = Path(paths[si])
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                x = pred.clone()
                x[:, :4] = scale_coords(img[si].shape[1:], x[:, :4], shapes[si][0], shapes[si][1])  # to original
                for *xyxy, conf, cls in x:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if dict['save_conf'] else (cls, *xywh)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # # W&B logging
            # if dict['plot'] and len(wandb_images) < log_imgs:
            #     box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
            #                  "class_id": int(cls),
            #                  "box_caption": "%s %.3f" % (names[int(cls)], conf),
            #                  "scores": {"class_score": conf},
            #                  "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
            #     boxes = {"predictions": {"box_data": box_data, "class_labels": names}}
            #     wandb_images.append(wandb.Image(img[si], boxes=boxes, caption=path.name))

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
                # pred = torch.from_numpy(np.array(pred)).to(device)
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    try:
                        pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices
                    except: # if pred's device = cpu
                        pred = torch.from_numpy(np.array(pred)).to(device)
                        pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break
                        # # Append detections
                        # detected_set = set()
                        # for j in (ious > iouv[0]).nonzero(as_tuple=False):
                        #     d = ti[i[j]]  # detected target
                        #     if d not in detected:
                        #         detected.append(d)
                        #         correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                        #         if len(detected) == nl:  # all targets already located in image
                        #             break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if dict['plot'] and batch_i < 3:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # filename
            plot_images(img, targets, paths, f, names)  # labels
            f = save_dir / f'test_batch{batch_i}_pred.jpg'
            plot_images(img, output_to_target(output, width, height), paths, f, names)  # predictions

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        # p, r, ap, f1, ap_class = ap_per_class(*stats)
        # p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=dict['plot'], fname=save_dir / 'precision-recall_curve.png')
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=dict['nclasses'])  # number of targets per class
    else:
        nt = torch.zeros(1)

    # # W&B logging
    # if dict['plot'] and dict['wandb']:
    #     wandb.log({"Images": wandb_images})
    #     wandb.log({"Validation": [wandb.Image(str(x), caption=x.name) for x in sorted(save_dir.glob('test*.jpg'))]})

    # Print results
    pf = '%20s' + '%12.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if verbose and dict['nclasses'] > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (dict['img_size'], dict['img_size'], dict['test_size'])  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Return results
    if not training:
        print('Results saved to %s' % save_dir)
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
        'img_size': 640, #Input image size. Must be a multiple of 32
        'batch_size': 32, #train batch size
        'test_size': 32, #test batch size

        # Data loader
        'rect': True,
        'aug': False,
        'img_format': '.jpeg',

        # test
        'nms_conf_t':0.001, #Confidence test threshold
        'study': False,

        #logs
        'save_txt': False,
        'save_conf': False,
        'plot': True,
        'wandb': False,

        # TODO: Image Format, , Comment, Weight_path, Img size, Aug., train/val set

        # PATH
        # 'weight_path': './runs/train/exp_RGB_BL_exp3_cuda/weights/last.pt',
        'weight_path': './runs/train/exp_IR_BL_640_100ms-from44RGB/weights/best_ap50.pt',
        # 'weight_path': './runs/train/aug/exp_IR320_MSMos_640for100_4/weights/last_299.pt',

        'task': 'test', # change to test only for the final test

        # large
        'test_path' : DATASET_PP_PATH + '/Train_Test_Split/test/',

        # mini
        # 'test_path' : DATASET_PP_PATH + '/mini_Train_Test_Split/test/',

        # # Day
        # 'test_path' : DATASET_PP_PATH + '/Train_Test_Split/train_Day/',
        # 'test_path' : DATASET_PP_PATH + '/Train_Test_Split/test_Day/',

        # # Night
        # 'test_path' : DATASET_PP_PATH + '/Train_Test_Split/train_Night/',
        # 'test_path' : DATASET_PP_PATH + '/Train_Test_Split/test_Night/',
     }

    hyp = {
        # test
        'iou_t': 0.65,  # IoU test threshold

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
        'mosaic': 0.0,
        'mixup': 0.0, #mix up probability
    }


    if not dict_['study']:  # run normally
        test(dict_, hyp, augment=dict_['aug']) # test augmentation

    # else:  # run over a range of settings and save/plot
    #     for weights in ['yolov4-csp.pt', 'yolov4-csp-x.pt']:
    #         f = 'study_%s_%s.txt' % (Path(opt.data).stem, Path(weights).stem)  # filename to save to
    #         x = list(range(320, 800, 64))  # x axis
    #         y = []  # y axis
    #         for i in x:  # img-size
    #             print('\nRunning %s point %s...' % (f, i))
    #             r, _, t = test(opt.data, weights, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json)
    #             y.append(r + t)  # results and times
    #         np.savetxt(f, y, fmt='%10.4g')  # save
    #     os.system('zip -r study.zip study_*.txt')
    #     # utils.general.plot_study_txt(f, x)  # plot
