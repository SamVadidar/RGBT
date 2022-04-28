from Fusion.models.models import *
import torch
import onnx


dict_ = {
    'device':'cuda', #Intialise device as cpu. Later check if cuda is avaialbel and change to cuda
    'device_num': '0',

    'anchors_g': [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]],
    'nclasses': 3, #Number of classes
    'names' : ['person', 'bicycle', 'car'],
    'img_size': 640, #Input image size. Must be a multiple of 32
    'batch_size':1,#train batch size
    'test_size': 1,#test batch size

    # Modules
    'H_attention_bc' : True, # entropy based att. before concat.
    'H_attention_ac' : True, # entropy based att. after concat.
    'spatial': True, # spatial attention off/on (channel is always by default on!)
    'weight_path': './runs/train/exp_RGBT640_500_HACBC_CS2/weights/best_val_loss_Ver2.pt', # best so far
    }

# initialize model
img_size = dict_['img_size']
img = torch.rand(1, 4, img_size, img_size, requires_grad=True)
model = Fused_Darknets(dict_, (img_size, img_size)).to("cpu")

# load weights
try:
    ckpt = torch.load(dict_['weight_path']) # load checkpoint
    if ckpt['epoch'] != -1: print('Saved @ epoch: ', ckpt['epoch'])
    
    ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(ckpt['model'], strict=False)
    print('Weights loaded')
except:
    raise ValueError('Check the "mode"/"modules" in your dict! Or maybe the Weight does not exist!')

# Dry run
model.eval()
torch_out = model(img[:, :3, :, :], img[:, 3:, :, :], augment=False)

# TorchScript export
try:
    print('\nStarting TorchScript export with torch %s...' % torch.__version__)
    f = dict_['weight_path'].replace('.pt', '.torchscript.pt')  # filename
    ts = torch.jit.trace(model, (img[:, :3, :, :], img[:, 3:, :, :]))
    ts.save(f)
    print('TorchScript export success, saved as %s' % f)
except Exception as e:
    print('TorchScript export failure: %s' % e)

# ONNX export
try:
    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    f = dict_['weight_path'].replace('.pt', '_1.onnx')  # filename
    # model.fuse()  # only for ONNX
    torch.onnx.export(model, (img[:, :3, :, :], img[:, 3:, :, :]), f, verbose=False, opset_version=12, input_names=['rgb', 'ir'],
                        output_names=['classes', 'boxes'] if torch_out is None else ['output'])

    # Checks
    onnx_model = onnx.load(f)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model
    # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
    print('ONNX export success, saved as %s' % f)
except Exception as e:
    print('ONNX export failure: %s' % e)

# Finish
print('\nExport complete. Visualize with https://github.com/lutzroeder/netron.')

# # Export the model
# torch.onnx.export(model,               # model being run
#                   (img[:, 1:, :, :], img[:, :1, :, :]), # model input (or a tuple for multiple inputs)
#                   "./RGBT.onnx",   # where to save the model (can be a file or file-like object)
#                   export_params=True,        # store the trained parameter weights inside the model file
#                   opset_version=10,          # the ONNX version to export the model to
#                   do_constant_folding=True,  # whether to execute constant folding for optimization
#                   input_names = ['input'],   # the model's input names
#                   output_names = ['output'], # the model's output names
#                   dynamic_axes={'input' : {1 : 'batch_size'},    # variable length axes
#                                 'output' : {1 : 'batch_size'}})