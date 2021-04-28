import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from yolo_backend import Darknet, clip_coords, scale_coords, non_max_suppression, xyxy2xywh, xywh2xyxy, Dataset, time_synchronized, box_iou, ap_per_class


def test(test_set, names, hyp, ckpt_path = None, model=None, txt_root = None, plot_all = False, break_no = 1000000):
    #Dataloader
    test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=hyp['test_size'],collate_fn=Dataset.collate_fn,shuffle=False)
    if model is None:
        model = Darknet(nclasses=hyp['nclasses'], anchors=np.array(hyp['anchors_g'])).to(hyp['device'])
        model.load_state_dict(torch.load(ckpt_path)['model'])
        print('Created pretrained model')
    model.eval()
    iouv = torch.linspace(0.5, 0.95, 10).to(hyp['device'])  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    p, r, f1, mp, mr, map50, m_ap, t0, t1, seen = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    stats, ap, ap_class = [], [], []
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    for batch_i, (img, targets, shapes, paths) in enumerate(tqdm(test_loader, desc=s)):
        if batch_i >= break_no:
            print("Either provide more img or reduce the defined 'break_no' value")
            break
        img = img.to(hyp['device'])/255.0; targets = targets.to(hyp['device'])
        # print(paths)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(hyp['device'])
        with torch.no_grad():
            #Run Model
            t = time_synchronized()
            inf_out, train_out = model(img)  # inference and training outputs
            t0 += time_synchronized() - t
            #Run NMS
            t = time_synchronized()
            output = non_max_suppression(inf_out, conf_thres=hyp['conf_t'], iou_thres=hyp['iou_t'])
            t1 += time_synchronized() - t
        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:] # filter out the labels of each image by index si
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target classes
            seen += 1

            if pred is None:
                if nl:
                    # niou(mAP@0.5:0.95) = 10 
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue#
            # Append to text file
            if txt_root is not None:
                txt_path = os.path.join(txt_root, paths[si].split(os.sep)[-1].replace('.jpg', '.txt'))
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                values = pred.clone()
                values[:, :4] = scale_coords(img[si].shape[1:], values[:, :4], shapes[si][0], shapes[si][1])  # to original
                for *xyxy, conf, clas in values:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    with open(txt_path, 'a') as f:
                        f.write(('%g ' * 5 + '\n') % (clas, *xywh))  # label format
            # Clip boxes to image bounds
            # args: (boxes, img.shape)
            clip_coords(pred, (height, width))
            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=hyp['device'])
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]
                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                for clas in torch.unique(tcls_tensor):
                    ti = (clas == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (clas == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

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
            original_img = Image.open(os.path.join(paths[si]))
            plt.rcParams['figure.figsize'] = (20,20)
            fig,ax = plt.subplots(1)
            ax.imshow(original_img)
            if pred is not None:
                boxes = pred
                boxes[:, :4] = scale_coords(img[si].shape[1:], boxes[:, :4], shapes[si][0], shapes[si][1])  # to original
                for i, box in enumerate(boxes.cpu()):
                    xmin = box[0]
                    ymin = box[1]
                    w = (box[2]-box[0])
                    h = (box[3]-box[1])
                    rect = patches.Rectangle((xmin,ymin),w,h,linewidth=2,edgecolor='r',facecolor='none')
                    ax.add_patch(rect)
                    ax.text(xmin, ymin, '%s %s'%(names[int(box[-1])],int(box[-2]*100)/100), fontsize = 12)
                plt.show()
            
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, m_ap = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=hyp['nclasses'])  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, m_ap))

    # Return results
    model.float()  # for training
    maps = np.zeros(hyp['nclasses']) + m_ap
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, m_ap), maps, t