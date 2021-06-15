import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # set environment variable

import torch
print(torch.cuda.current_device())
a = torch.zeros((5, 12)).to('cuda:0')
