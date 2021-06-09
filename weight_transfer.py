import torch
import numpy as np

# dict = torch.load("./yolo_pre_3c.pt")['model']
dict = torch.load("./init.pt")

els=[]
keys = []
for key in dict.keys():
    keys.append(key)
    els.append(torch.numel(dict[key]))
a = np.array(list(dict.keys()))
b = np.array(els)
print(a)
print(b)