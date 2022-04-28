import onnxruntime
import numpy as np

from PIL import Image
import torch
from Fusion.utils.datasets import create_dataloader
from FLIR_PP.arg_parser import DATASET_PP_PATH
from tqdm import tqdm
from Fusion.utils.torch_utils import select_device, time_synchronized
from Fusion.utils.general import box_iou, non_max_suppression, xywh2xyxy, clip_coords, increment_path
from Fusion.utils.metrics import ap_per_class
from Fusion.utils.plots import plot_images, output_to_target
from pathlib import Path


def to_numpy(tensor):
    # if type(tensor) == list:
    #     tensor = torch.as_tensor(tensor)
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


dict_ = {
    'device':'cuda', #Intialise device as cpu. Later check if cuda is avaialbel and change to cuda
    'device_num': '0',

    # Kmeans on COCO
    'anchors_g': [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]],
    'nclasses': 3, #Number of classes
    'names' : ['person', 'bicycle', 'car'],
    'img_size': 640, #Input image size. Must be a multiple of 32
    'img_format': '.jpg',
    'batch_size':1,#train batch size
    'test_size': 1,#test batch size

    # test
    'nms_conf_t':0.001, #Confidence test threshold
    'nms_merge': True,

    # Data loader
    'rect': False,
    'aug': False,
    'mode': 'fusion', #Options: ir / rgb / fusion
    'comment': '',
    'compare': True,

    # Modules
    'H_attention_bc' : True, # entropy based att. before concat.
    'H_attention_ac' : True, # entropy based att. after concat.
    'spatial': True, # spatial attention off/on (channel is always by default on!)


    'weight_path': './runs/train/exp_RGBT640_500_HACBC_CS2/weights/best_val_loss_Ver2.pt', # best so far
    'ONNX_Model': './runs/train/exp_RGBT640_500_HACBC_CS2/weights/best_val_loss_Ver2_1.onnx',
    'test_path' : DATASET_PP_PATH + '/mini_Train_Test_Split/SingleImg/',
    'plot': True,
    'save_txt': False,
}

dict_['comment'] = dict_['weight_path'][(dict_['weight_path'].find('_')+1):]
dict_['comment'] = dict_['comment'][:dict_['comment'].find('/')]

# Directories
save_dir = Path(increment_path(('./runs/test/exp'+dict_['comment']), exist_ok=False, sep='_'))  # increment run
(save_dir / 'labels' if dict_['save_txt'] else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

if dict_['compare']:
    from Fusion.models.models import *

    device = select_device(device=dict_['device_num'], batch_size=dict_['batch_size'])
    img_size = dict_['img_size']
    model = Fused_Darknets(dict_, (img_size, img_size)).to(device)

    # load model
    try:
        ckpt = torch.load(dict_['weight_path']) # load checkpoint
        if ckpt['epoch'] != -1: print('Saved @ epoch: ', ckpt['epoch'])
        
        ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(ckpt['model'], strict=False)
    except:
        raise ValueError('Check the "mode"/"modules" in your dict! Or maybe the Weight does not exist!')

    model.eval()

dataloader = create_dataloader(dict_['test_path'] , dict_['img_size'], dict_['batch_size'], 64,
                                hyp=None, augment=dict_['aug'], pad=0.5, rect=dict_['rect'],
                                img_format=dict_['img_format'], mode = dict_['mode'])[0] # grid_size=32

device = select_device(device=dict_['device_num'], batch_size=dict_['batch_size'])
seen = 0
iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
niou = iouv.numel()
names = dict_['names']
s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
loss = torch.zeros(3, device=device)
jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
    img = img.to(device, non_blocking=True)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    targets = targets.to(device)
    nb, _, height, width = img.shape  # batch size, channels, height, width
    whwh = torch.Tensor([width, height, width, height]).to(device)

    #define the priority order for the execution providers
    # prefer CUDA Execution Provider over CPU Execution Provider
    EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    ort_session = onnxruntime.InferenceSession(dict_['ONNX_Model'], providers=EP_list)
    # Disable gradients
    with torch.no_grad():
        input_name = [input.name for input in ort_session.get_inputs()]
        output_name = [output.name for output in ort_session.get_outputs()]
        print("inputs name:",input_name,"|| outputs name:",output_name)

        # Run model
        t = time_synchronized()
        # compute ONNX Runtime output prediction
        if dict_['compare']:
            torch_out, train_out = model(img[:, 1:, :, :], img[:, :1, :, :], augment=False) # (RGB, IR)

        if ort_session.get_providers()[0] == EP_list[1]:
            ort_outs = ort_session.run(output_name, {input_name[0]:to_numpy(img[:, 1:, :, :]),
                                                     input_name[1]:to_numpy(img[:, :1, :, :])})
        else:
            print(ort_session.get_providers()[0])
            ort_outs = ort_session.run(output_name, {input_name[0]:img[:, 1:, :, :],
                                                     input_name[1]:img[:, :1, :, :]})                                        
        inf_out = ort_outs[0]
        t0 += time_synchronized() - t

        # Run NMS
        t = time_synchronized()
        inf_out = torch.from_numpy(inf_out).to(device=device)
        output = non_max_suppression(inf_out, conf_thres=dict_['nms_conf_t'], iou_thres=0.5, merge=dict_['nms_merge'])
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

        # Append statistics (correct, conf, pcls, tcls)
        stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    # Plot images
    if dict_['plot'] and batch_i < 10:
        # f = save_dir / f'test_batch{batch_i}_labels.jpg'  # filename
        f = str(save_dir) + f'/test_batch{batch_i}_labels' + dict_['img_format']  # filename
        plot_images(img, targets, paths, f, names)  # labels
        # f = save_dir / f'test_batch{batch_i}_pred.jpg'
        f = str(save_dir) + f'/test_batch{batch_i}_pred' + dict_['img_format']  # filename
        plot_images(img, output_to_target(output, width, height), paths, f, names)  # predictions

# Compute statistics
stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
if len(stats) and stats[0].any():
    # p, r, ap, f1, ap_class = ap_per_class(*stats)
    # p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
    p, r, ap, f1, ap_class = ap_per_class(*stats, plot=dict_['plot'], fname=save_dir / 'precision-recall_curve.png')
    p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
    mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(np.int64), minlength=dict_['nclasses'])  # number of targets per class
else:
    nt = torch.zeros(1)

# Print results
pf = '%20s' + '%12.3g' * 6  # print format
print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

# Print results per class
if dict_['nclasses'] > 1 and len(stats):
    for i, c in enumerate(ap_class):
        print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

# Print speeds
t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (dict_['img_size'], dict_['img_size'], dict_['test_size'])  # tuple
print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

# Return results
print('Results saved to %s' % save_dir)


# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
print("Exported model has been tested with ONNXRuntime, and the result looks good!")