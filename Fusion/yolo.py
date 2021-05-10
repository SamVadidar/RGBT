# -*- coding: utf-8 -*-
"""
Adapted From: WongKinYiu and Gokulesh Danapal
https://github.com/WongKinYiu/ScaledYOLOv4
https://github.com/gokulesh-danapal
"""
import torch
import numpy as np
from torchsummary import summary

class Mish(torch.nn.Module):
    def __init__(self):
        super(Mish,self).__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x
    
class CBM(torch.nn.Module):
    def __init__(self,in_filters, out_filters, kernel_size, stride):
        super(CBM,self).__init__()                               
        self.conv = torch.nn.Conv2d(in_channels=in_filters,out_channels=out_filters,kernel_size=kernel_size,stride=stride,padding=kernel_size//2,bias=False)   
        self.batchnorm = torch.nn.BatchNorm2d(num_features=out_filters,momentum=0.03, eps=1E-4)
        self.act = Mish()
    def forward(self,x):
        return self.act(self.batchnorm(self.conv(x)))
        
class ResUnit(torch.nn.Module):
    def __init__(self, filters, first = False):
        super(ResUnit, self).__init__()
        if first:
            self.out_filters = filters//2
        else:
            self.out_filters = filters         
        self.resroute= torch.nn.Sequential(CBM(filters, self.out_filters, kernel_size=1, stride=1),
                                                    CBM(self.out_filters, filters, kernel_size=3, stride=1))       
    def forward(self, x):
        shortcut = x
        x = self.resroute(x)
        return x+shortcut

class CSP(torch.nn.Module):
    def __init__(self, filters, nblocks):
        super(CSP,self).__init__()
        self.skip = CBM(in_filters=filters,out_filters=filters//2,kernel_size=1,stride=1)
        self.route_list = torch.nn.ModuleList()
        self.route_list.append(CBM(in_filters=filters,out_filters=filters//2,kernel_size=1,stride=1))
        for block in range(nblocks):
            self.route_list.append(ResUnit(filters=filters//2))
        self.route_list.append(CBM(in_filters=filters//2,out_filters=filters//2,kernel_size=1,stride=1))                                         
        self.last = CBM(in_filters=filters,out_filters=filters,kernel_size=1,stride=1)
        
    def forward(self,x):
        shortcut = self.skip(x)
        for block in self.route_list:
            x = block(x)
        x = torch.cat((x,shortcut),dim = 1)
        return self.last(x)

class SPP(torch.nn.Module):
    def __init__(self,filters):
        super(SPP,self).__init__()
        self.maxpool5 = torch.nn.MaxPool2d(kernel_size=5,stride=1,padding = 5//2)
        self.maxpool9 = torch.nn.MaxPool2d(kernel_size=9,stride=1,padding = 9//2)
        self.maxpool13 = torch.nn.MaxPool2d(kernel_size=13,stride=1,padding = 13//2)
    def forward(self,x):
        x5 = self.maxpool5(x)
        x9 = self.maxpool9(x)
        x13 = self.maxpool13(x)
        return torch.cat((x13,x9,x5,x),dim=1)
            
class rCSP(torch.nn.Module):
    def __init__(self,filters,spp_block = False):
        super(rCSP,self).__init__()
        self.include_spp = spp_block
        if self.include_spp:
            self.in_filters = filters*2
        else:
            self.in_filters = filters
        self.skip = CBM(in_filters=self.in_filters,out_filters=filters,kernel_size=1,stride=1)
        self.module_list = torch.nn.ModuleList()
        self.module_list.append(torch.nn.Sequential(CBM(in_filters=self.in_filters,out_filters=filters,kernel_size=1,stride=1),
                                CBM(in_filters=filters,out_filters=filters,kernel_size=3,stride=1),
                                CBM(in_filters=filters,out_filters=filters,kernel_size=1,stride=1)))
        if self.include_spp:
            self.module_list.append(torch.nn.Sequential(SPP(filters=filters),
                                    CBM(in_filters=filters*4,out_filters=filters,kernel_size=1,stride=1)))
        self.module_list.append(CBM(in_filters=filters,out_filters=filters,kernel_size=3,stride=1))
        self.last = CBM(in_filters=filters*2,out_filters=filters,kernel_size=1,stride=1)
    def forward(self,x):
        shortcut = self.skip(x)
        for block in self.module_list:
            x = block(x)
        x = torch.cat((x,shortcut),dim=1)
        x = self.last(x)
        return x 
    
def up(filters):
        return torch.nn.Sequential(CBM(in_filters=filters,out_filters=filters//2,kernel_size=1,stride=1),
                                        torch.nn.Upsample(scale_factor=2))

class YOLOLayer(torch.nn.Module):
    def __init__(self, anchors, nc, stride):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        #self.index = yolo_index  # index of this layer in layers
        #self.layers = layers  # model output layer indices
        self.stride = stride  # layer stride
        #self.nl = len(layers)  # number of output layers (3)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs (85)
        self.nx, self.ny, self.ng = 0, 0, 0  # initialize number of x, y gridpoints
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)

    def create_grids(self, ng=(13, 13), device='cpu'):
        self.nx, self.ny = ng  # x and y grid size
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offsets
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device), torch.arange(self.nx, device=device)])
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p):
        bs, _, ny, nx = p.shape  # bs, 255, 13, 13
        if (self.nx, self.ny) != (nx, ny):
            self.create_grids((nx, ny), p.device)
        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p
        else:  # inference
            io = p.sigmoid()
            io[..., :2] = (io[..., :2] * 2. - 0.5 + self.grid)
            io[..., 2:4] = (io[..., 2:4] * 2) ** 2 * self.anchor_wh
            io[..., :4] *= self.stride
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]

        
class Backbone(torch.nn.Module):
    def __init__(self):
        super(Backbone,self).__init__()
        self.main3 = torch.nn.Sequential(CBM(in_filters=3,out_filters=32,kernel_size=3,stride=1),
                                        CBM(in_filters=32,out_filters=64,kernel_size=3,stride=2),
                                        ResUnit(filters = 64, first= True),
                                        CBM(in_filters=64,out_filters=128,kernel_size=3,stride=2),
                                        CSP(filters=128,nblocks = 2), 
                                        CBM(in_filters=128,out_filters=256,kernel_size=3,stride=2),
                                        CSP(filters=256,nblocks = 8))
        self.main4 = torch.nn.Sequential(CBM(in_filters=256,out_filters=512,kernel_size=3,stride=2),
                                        CSP(filters=512,nblocks = 8))
        self.main5 = torch.nn.Sequential(CBM(in_filters=512,out_filters=1024,kernel_size=3,stride=2),
                                        CSP(filters=1024,nblocks = 4))
    def forward(self,x):
        x3 = self.main3(x)
        x4 = self.main4(x3)
        x5 = self.main5(x4)
        return (x3,x4,x5)
    
class Neck(torch.nn.Module):
    def __init__(self):
        super(Neck,self).__init__()
        self.main5 = rCSP(512,spp_block=True)
        self.up5 = up(512)
        self.conv1 = CBM(in_filters=512,out_filters=256,kernel_size=1,stride=1)
        self.conv2 = CBM(in_filters=512,out_filters=256,kernel_size=1,stride=1)
        self.main4 = rCSP(256)
        self.up4 = up(256)
        self.conv3 = CBM(in_filters=256,out_filters=128,kernel_size=1,stride=1)
        self.conv4 = CBM(in_filters=256,out_filters=128,kernel_size=1,stride=1)
        self.main3 = rCSP(128)
    def forward(self,x):
        x3 = x[0]; x4 = x[1]; x5= x[2];
        x5 = self.main5(x5)
        x4 = self.main4(self.conv2(torch.cat((self.conv1(x4),self.up5(x5)),dim=1)))
        x3 = self.main3(self.conv4(torch.cat((self.conv3(x3),self.up4(x4)),dim=1)))
        return (x3,x4,x5)
    
class Head(torch.nn.Module):
    def __init__(self,nclasses):
        super(Head,self).__init__()
        self.last_layers = 3*(4+1+nclasses)
        self.last3 = CBM(in_filters=128,out_filters=256,kernel_size=3,stride=1)
        self.final3 = torch.nn.Conv2d(in_channels=256,out_channels=self.last_layers,kernel_size=1,stride=1,bias=True) 
        
        self.conv1 = CBM(in_filters=128,out_filters=256,kernel_size=3,stride=2)
        self.conv2 = CBM(in_filters=512,out_filters=256,kernel_size=1,stride=1)
        self.main4 = rCSP(256)
        self.last4 = CBM(in_filters=256,out_filters=512,kernel_size=3,stride=1)
        self.final4 = torch.nn.Conv2d(in_channels=512,out_channels=self.last_layers,kernel_size=1,stride=1,bias=True)
        
        self.conv3 = CBM(in_filters=256,out_filters=512,kernel_size=3,stride=2)
        self.conv4 = CBM(in_filters=1024,out_filters=512,kernel_size=1,stride=1)
        self.main5 = rCSP(512)
        self.last5 = CBM(in_filters=512,out_filters=1024,kernel_size=3,stride=1)
        self.final5 = torch.nn.Conv2d(in_channels=1024,out_channels=self.last_layers,kernel_size=1,stride=1,bias=True)
    def forward(self,x):
        x3 = x[0]; x4 = x[1]; x5= x[2];
        y3 = self.final3(self.last3(x3))
        x4 = self.main4(self.conv2(torch.cat((self.conv1(x3),x4),dim=1)))
        y4 = self.final4(self.last4(x4))
        x5 = self.main5(self.conv4(torch.cat((self.conv3(x4),x5),dim=1)))
        y5 = self.final5(self.last5(x5))
        return y3,y4,y5

class Darknet(torch.nn.Module):
    def __init__(self,nclasses,anchors):
        super(Darknet,self).__init__()
        self.nclasses = nclasses
        self.anchors = anchors
        self.backbone = Backbone()
        self.neck = Neck()
        self.head = Head(self.nclasses)
        self.yolo3 = YOLOLayer(self.anchors[0:3], self.nclasses, stride = 8)
        self.yolo4 = YOLOLayer(self.anchors[3:6], self.nclasses, stride = 16)
        self.yolo5 = YOLOLayer(self.anchors[6:9], self.nclasses, stride = 32)
    def forward(self,x):
        y3,y4,y5 = self.head(self.neck(self.backbone(x)))
        y3 = self.yolo3(y3)
        y4 = self.yolo4(y4)
        y5 = self.yolo5(y5)
        yolo_out = [y3,y4,y5]
        if self.training:
            return yolo_out
        else:  # inference or test
            x, p = zip(*yolo_out)  # inference output, training output
            x = torch.cat(x, 1)  # cat yolo outputs
            return x, p


if __name__ == '__main__':
    anchors_g = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]]
    model = Darknet(nclasses=80, anchors=np.array(anchors_g)).to('cuda')
    summary(model, torch.zeros((2, 3, 416, 416)), depth=10)