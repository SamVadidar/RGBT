import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
from torch._C import device
import Conv_ONNX2TensorRT as eng
import pycuda.autoinit 

def allocate_buffers(engine, batch_size, data_type):

   """
   This is the function to allocate buffers for input and output in the device
   Args:
      engine : The path to the TensorRT engine. 
      batch_size : The batch size for execution time.
      data_type: The type of the data for input and output, for example trt.float32. 
   
   Output:
      h_input_1: Input in the host.
      d_input_1: Input in the device. 
      h_output_1: Output in the host. 
      d_output_1: Output in the device. 
      stream: CUDA stream.

   """

   # Determine dimensions and create page-locked memory buffers (which won't be swapped to disk) to hold host inputs/outputs.
   h_input_rgb = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(data_type)) # 1*3*640*640*(float32=4)
   h_input_ir = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(data_type)) # 1*1*640*640*(float32=4)
   h_output_s1 = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(2)), dtype=trt.nptype(data_type)) # 1*3*80*80*8*(float32=4)
   h_output_s2 = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(3)), dtype=trt.nptype(data_type)) # 1*3*40*40*8*(float32=4)
   h_output_s3 = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(4)), dtype=trt.nptype(data_type)) # 1*3*20*20*8*(float32=4)
   h_output = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(5)), dtype=trt.nptype(data_type)) # 1*25200*8*(float32=4)

   # print(engine.get_binding_shape(0),
   #       engine.get_binding_shape(1),
   #       engine.get_binding_shape(2),
   #       engine.get_binding_shape(3),
   #       engine.get_binding_shape(4),
   #       engine.get_binding_shape(5))

   # Allocate device memory for inputs and outputs.
   d_input_rgb = cuda.mem_alloc(h_input_rgb.nbytes)
   d_input_ir = cuda.mem_alloc(h_input_ir.nbytes)
   d_output_s1 = cuda.mem_alloc(h_output_s1.nbytes)
   d_output_s2 = cuda.mem_alloc(h_output_s2.nbytes)
   d_output_s3 = cuda.mem_alloc(h_output_s3.nbytes)
   d_output = cuda.mem_alloc(h_output.nbytes)

   # Create a stream in which to copy inputs/outputs and run inference.
   stream = cuda.Stream()
   return h_input_rgb, d_input_rgb, h_input_ir, d_input_ir, h_output_s1, d_output_s1, h_output_s2, d_output_s2, h_output_s3, d_output_s3, h_output, d_output, stream

# class HostDeviceMem(object):
#     def __init__(self, host_mem, device_mem):
#         self.host = host_mem
#         self.device = device_mem

#     def __str__(self):
#         return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

#     def __repr__(self):
#         return self.__str__()

# # Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
# def allocate_buffers(engine, batch_size=-1):
#    inputs = []
#    outputs = []
#    bindings = []
#    stream = cuda.Stream()
#    for binding in engine:
#       # pdb.set_trace()
#       size = trt.volume(engine.get_binding_shape(binding)) * batch_size
#       dtype = trt.nptype(engine.get_binding_dtype(binding))
#       # Allocate host and device buffers
#       host_mem = cuda.pagelocked_empty(size, dtype)
#       device_mem = cuda.mem_alloc(host_mem.nbytes)
#       # Append the device buffer to device bindings.
#       bindings.append(int(device_mem))
#       # Append to the appropriate list.
#       if engine.binding_is_input(binding):
#          inputs.append(HostDeviceMem(host_mem, device_mem))
#          print(f"input: shape:{engine.get_binding_shape(binding)} dtype:{engine.get_binding_dtype(binding)}")
#       else:
#          outputs.append(HostDeviceMem(host_mem, device_mem))
#          print(f"output: shape:{engine.get_binding_shape(binding)} dtype:{engine.get_binding_dtype(binding)}")
#    return inputs, outputs, bindings, stream
    
def load_images_to_buffer(pics, pagelocked_buffer):
   if pics.is_cuda:
      pics = pics.cpu()
   preprocessed = np.asarray(pics).ravel()
   np.copyto(pagelocked_buffer, preprocessed) 

def do_inference(engine, img, h_input_rgb, d_input_rgb, h_input_ir, d_input_ir, h_output_s1, d_output_s1, h_output_s2, d_output_s2, h_output_s3, d_output_s3, h_output, d_output, stream, output_shape):
   """
   This is the function to run the inference
   Args:
      engine : Path to the TensorRT engine 
      pics_1 : Input images to the model.  
      h_input_1: Input in the host         
      d_input_1: Input in the device 
      h_output_1: Output in the host 
      d_output_1: Output in the device 
      stream: CUDA stream
      batch_size : Batch size for execution time
      height: Height of the output image
      width: Width of the output image
   
   Output:
      The list of output images
   """

   load_images_to_buffer(img[:,1:,:,:], h_input_rgb)
   load_images_to_buffer(img[:,:1,:,:], h_input_ir)

   with engine.create_execution_context() as context:
      # Transfer input data to the GPU
      cuda.memcpy_htod_async(d_input_rgb, h_input_rgb, stream)
      cuda.memcpy_htod_async(d_input_ir, h_input_ir, stream)

      # # finding the input/output names
      # for i in range(6):
      #    print(engine.get_binding_name(i))

      # input_rgb_idx = engine['images'] # RGB
      # input_ir_idx = engine['input.147'] # IR
      # output_idx = engine['output'] # Output
      # print(input_rgb_idx, input_ir_idx, output_idx)

      # print(int(d_output))
      # print(engine.get_binding_shape(0))

      # Run inference
      context.profiler = trt.Profiler()
      context.execute(batch_size=1, bindings=[int(d_input_rgb), int(d_input_ir), int(d_output), int(d_output_s1), int(d_output_s2), int(d_output_s3)])

      # Transfer predictions back from the GPU
      cuda.memcpy_dtoh_async(h_output, d_output, stream)
      # Synchronize the stream
      stream.synchronize()

      # Return the host output
      out = h_output.reshape((output_shape[0],output_shape[1], output_shape[2])) # bs*25200*8 as yolo output
      return out

def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
   # Transfer input data to the GPU.
   [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
   # Run inference.
   context.execute_async(
      batch_size=batch_size, bindings=bindings, stream_handle=stream.handle
   )
   # Transfer predictions back from the GPU.
   [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
   # Synchronize the stream
   stream.synchronize()
   # Return only the host outputs.
   return [out.host for out in outputs]

if __name__ == "__main__":
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

      # Modules
      'H_attention_bc' : True, # entropy based att. before concat.
      'H_attention_ac' : True, # entropy based att. after concat.
      'spatial': True, # spatial attention off/on (channel is always by default on!)


      'weight_path': './runs/train/exp_RGBT640_500_HACBC_CS2/weights/best_val_loss_Ver2.pt', # best so far
      'model_RT_path': './RGBT_new_4Channel_1.plan',
      'test_path' : DATASET_PP_PATH + '/mini_Train_Test_Split/SingleImg/',
      'plot': True,
      'save_txt': False,
   }

   dict_['comment'] = dict_['weight_path'][(dict_['weight_path'].find('_')+1):]
   dict_['comment'] = dict_['comment'][:dict_['comment'].find('/')]

   # Directories
   save_dir = Path(increment_path(('./runs/test/exp'+dict_['comment']), exist_ok=False, sep='_'))  # increment run
   (save_dir / 'labels' if dict_['save_txt'] else save_dir).mkdir(parents=True, exist_ok=True)  # make dir


   TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
   trt_runtime = trt.Runtime(TRT_LOGGER)
   serialized_plan_fp32 = dict_["model_RT_path"]

   dataloader = create_dataloader(dict_['test_path'] , dict_['img_size'], dict_['batch_size'], 64,
                                  hyp=None, augment=dict_['aug'], pad=0.5, rect=dict_['rect'],
                                  img_format=dict_['img_format'], mode = dict_['mode'])[0] # grid_size=32

   engine = eng.load_engine(trt_runtime, serialized_plan_fp32)
   h_input_rgb, d_input_rgb, h_input_ir, d_input_ir, h_output_s1, d_output_s1, h_output_s2, d_output_s2, h_output_s3, d_output_s3, h_output, d_output, stream = allocate_buffers(engine, dict_['batch_size'], trt.float32)
   
   # Execution context is needed for inference
   # context = engine.create_execution_context()
   # This allocates memory for network inputs/outputs on both CPU and GPU
   # inputs, outputs, bindings, stream = allocate_buffers(engine)

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
      # img = img.to(device, non_blocking=True)
      img = img.float()  # uint8 to fp16/32
      img /= 255.0  # 0 - 255 to 0.0 - 1.0
      targets = targets.to(device)
      nb, _, height, width = img.shape  # batch size, channels, height, width
      whwh = torch.Tensor([width, height, width, height]).to(device)

      # Disable gradients
      with torch.no_grad():
         # Run model
         t = time_synchronized()
         inf_out = do_inference(engine, img, h_input_rgb, d_input_rgb, h_input_ir, d_input_ir, h_output_s1, d_output_s1, h_output_s2, d_output_s2, h_output_s3, d_output_s3, h_output, d_output, stream, output_shape=[dict_['batch_size'],25200,8])
         # inf_out = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
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
   # model.float()  # for training
   # maps = np.zeros(dict_['nclasses']) + map
   # for i, c in enumerate(ap_class):
   #    maps[c] = ap[i]

   ##############################################################################################################

   # color_map = np.array(out, dtype=np.float32, order='C').reshape((WIDTH, HEIGHT), order='C')
   # colorImage_trt = Image.fromarray(color_map.astype(np.uint8))
   # colorImage_trt.save("trt_output.png")

   # semantic_model = keras.models.load_model('/path/to/semantic_segmentation.hdf5')
   # out_keras= semantic_model.predict(im.reshape(-1, 3, HEIGHT, WIDTH))

   # out_keras = color_map(out_keras)
   # colorImage_k = Image.fromarray(out_keras.astype(np.uint8))
   # colorImage_k.save(“keras_output.png”)