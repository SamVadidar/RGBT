from Fusion.utils.google_utils import *
from Fusion.utils.layers import *
from Fusion.utils import torch_utils
import matplotlib.pyplot as plt

ONNX_EXPORT = False


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
    def __init__(self, filters):
        super(SPP, self).__init__()
        self.maxpool5 = torch.nn.MaxPool2d(kernel_size=5,stride=1,padding = 5//2)
        self.maxpool9 = torch.nn.MaxPool2d(kernel_size=9,stride=1,padding = 9//2)
        self.maxpool13 = torch.nn.MaxPool2d(kernel_size=13,stride=1,padding = 13//2)
    def forward(self,x):
        x5 = self.maxpool5(x)
        x9 = self.maxpool9(x)
        x13 = self.maxpool13(x)
        return torch.cat((x13,x9,x5,x),dim=1)


class rCSP(torch.nn.Module):
    def __init__(self, filters, spp_block = False):
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

    def forward(self, x):
        shortcut = self.skip(x)
        for block in self.module_list:
            x = block(x)
        x = torch.cat((x, shortcut), dim=1)
        x = self.last(x)
        return x


class Fused_Backbone(torch.nn.Module):
    def __init__(self, H_att_bc=False, H_att_ac=False, spatial=False):
        super(Fused_Backbone,self).__init__()

        self.H_attention_bc = H_att_bc # EntropyBasedAttention before concat
        self.H_attention_ac = H_att_ac # EntropyBasedAttention after concat

        self.main3_rgb = torch.nn.Sequential(CBM(in_filters=3,out_filters=32,kernel_size=3,stride=1),
                                        CBM(in_filters=32,out_filters=64,kernel_size=3,stride=2),
                                        ResUnit(filters = 64, first= True),
                                        CBM(in_filters=64,out_filters=128,kernel_size=3,stride=2),
                                        CSP(filters=128,nblocks = 2),
                                        CBM(in_filters=128,out_filters=256,kernel_size=3,stride=2),
                                        CSP(filters=256,nblocks = 8))
        self.main4_rgb = torch.nn.Sequential(CBM(in_filters=256,out_filters=512,kernel_size=3,stride=2),
                                        CSP(filters=512,nblocks = 8))
        self.main5_rgb = torch.nn.Sequential(CBM(in_filters=512,out_filters=1024,kernel_size=3,stride=2),
                                        CSP(filters=1024,nblocks = 4))

        self.main3_ir = torch.nn.Sequential(CBM(in_filters=1,out_filters=32,kernel_size=3,stride=1),
                                        CBM(in_filters=32,out_filters=64,kernel_size=3,stride=2),
                                        ResUnit(filters = 64, first= True),
                                        CBM(in_filters=64,out_filters=128,kernel_size=3,stride=2),
                                        CSP(filters=128,nblocks = 2),
                                        CBM(in_filters=128,out_filters=256,kernel_size=3,stride=2),
                                        CSP(filters=256,nblocks = 8))
        self.main4_ir = torch.nn.Sequential(CBM(in_filters=256,out_filters=512,kernel_size=3,stride=2),
                                        CSP(filters=512,nblocks = 8))
        self.main5_ir = torch.nn.Sequential(CBM(in_filters=512,out_filters=1024,kernel_size=3,stride=2),
                                        CSP(filters=1024,nblocks = 4))

        if self.H_attention_bc:
            self.ebam_x3_bc = EBAM(256, spatial=spatial)
            self.ebam_x4_bc = EBAM(512, spatial=spatial)
            self.ebam_x5_bc = EBAM(1024, spatial=spatial)

        if self.H_attention_ac:
            self.ebam_x3_ac = EBAM(512, spatial=spatial)
            self.ebam_x4_ac = EBAM(1024, spatial=spatial)
            self.ebam_x5_ac = EBAM(2048, spatial=spatial)

        self.f_x3_Conv2d = torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, bias=False) # torch.Size([1, 512, 80, 80])
        self.f_x4_Conv2d = torch.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, bias=False)
        self.f_x5_Conv2d = torch.nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1, stride=1, bias=False)

    def forward(self, rgb, ir):
        if self.H_attention_bc:
            rgb_x3 = self.ebam_x3_bc(self.main3_rgb(rgb))
            rgb_x4 = self.ebam_x4_bc(self.main4_rgb(rgb_x3))
            rgb_x5 = self.ebam_x5_bc(self.main5_rgb(rgb_x4))

            ir_x3 = self.ebam_x3_bc(self.main3_ir(ir))
            ir_x4 = self.ebam_x4_bc(self.main4_ir(ir_x3))
            ir_x5 = self.ebam_x5_bc(self.main5_ir(ir_x4))
        else:
            rgb_x3 = self.main3_rgb(rgb)
            rgb_x4 = self.main4_rgb(rgb_x3)
            rgb_x5 = self.main5_rgb(rgb_x4)

            ir_x3 = self.main3_ir(ir)
            ir_x4 = self.main4_ir(ir_x3)
            ir_x5 = self.main5_ir(ir_x4)

        f_x3 = torch.cat((rgb_x3, ir_x3), dim=1) # torch.Size([1, 512, 80, 80])
        f_x4 = torch.cat((rgb_x4, ir_x4), dim=1) # torch.Size([1, 1024, 40, 40])
        f_x5 = torch.cat((rgb_x5, ir_x5), dim=1) # torch.Size([1, 2048, 20, 20])

        if self.H_attention_ac:
            f_x3 = self.ebam_x3_ac(f_x3)
            f_x4 = self.ebam_x4_ac(f_x4)
            f_x5 = self.ebam_x5_ac(f_x5)

            f_x3 = self.f_x3_Conv2d(f_x3)
            f_x4 = self.f_x4_Conv2d(f_x4)
            f_x5 = self.f_x5_Conv2d(f_x5)
        else:
            f_x3 = self.f_x3_Conv2d(f_x3)
            f_x4 = self.f_x4_Conv2d(f_x4)
            f_x5 = self.f_x5_Conv2d(f_x5)

        return (f_x3, f_x4, f_x5)


class Backbone(torch.nn.Module):
    def __init__(self, mode, attention=False, H_attention=False, spatial=False):
        super(Backbone,self).__init__()
        self.attention = attention
        self.H_attention = H_attention
        if mode == 'rgb':
            self.main3 = torch.nn.Sequential(CBM(in_filters=3,out_filters=32,kernel_size=3,stride=1),
                                            CBM(in_filters=32,out_filters=64,kernel_size=3,stride=2),
                                            ResUnit(filters = 64, first= True),
                                            CBM(in_filters=64,out_filters=128,kernel_size=3,stride=2),
                                            CSP(filters=128,nblocks = 2),
                                            CBM(in_filters=128,out_filters=256,kernel_size=3,stride=2),
                                            CSP(filters=256,nblocks = 8))
        elif mode == 'ir':
            self.main3 = torch.nn.Sequential(CBM(in_filters=1,out_filters=32,kernel_size=3,stride=1),
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

        if self.attention:
            self.ebam_x3 = EBAM(256, spatial=True)
            self.ebam_x4 = EBAM(512, spatial=True)
            self.ebam_x5 = EBAM(1024, spatial=True)
        if self.H_attention:
            self.ebam_x3 = EBAM(256, spatial=spatial)
            self.ebam_x4 = EBAM(512, spatial=spatial)
            self.ebam_x5 = EBAM(1024, spatial=spatial)

    def forward(self,x):
        if self.attention or self.H_attention:
            x3 = self.ebam_x3(self.main3(x))
            x4 = self.ebam_x4(self.main4(x3))
            x5 = self.ebam_x5(self.main5(x4))
        else:
            x3 = self.main3(x)
            x4 = self.main4(x3)
            x5 = self.main5(x4)
        return (x3,x4,x5)


def up(filters):
        return torch.nn.Sequential(CBM(in_filters=filters,out_filters=filters//2,kernel_size=1,stride=1),
                                        torch.nn.Upsample(scale_factor=2))


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


class YOLOLayer(torch.nn.Module):
    def __init__(self, anchors, nc, img_size, stride):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.stride = stride  # layer stride
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs (85)
        self.nx, self.ny, self.ng = 0, 0, 0  # initialize number of x, y gridpoints
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)

        if ONNX_EXPORT:
            self.training = False
            self.create_grids((img_size[1] // stride, img_size[0] // stride))  # number x, y grid points

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

    def forward(self, p): # , out
        ASFF = False  # https://arxiv.org/abs/1911.09516
        if ASFF:
            raise ValueError('ASFF is not supported')
            # # org below
            # i, n = self.index, self.nl  # index in layers, number of layers
            # p = out[self.layers[i]]
            # bs, _, ny, nx = p.shape  # bs, 255, 13, 13
            # if (self.nx, self.ny) != (nx, ny):
            #     self.create_grids((nx, ny), p.device)
            # # outputs and weights
            # # w = F.softmax(p[:, -n:], 1)  # normalized weights
            # w = torch.sigmoid(p[:, -n:]) * (2 / n)  # sigmoid weights (faster)
            # # w = w / w.sum(1).unsqueeze(1)  # normalize across layer dimension
            # # weighted ASFF sum
            # p = out[self.layers[i]][:, :-n] * w[:, i:i + 1]
            # for j in range(n):
            #     if j != i:
            #         p += w[:, j:j + 1] * \
            #              F.interpolate(out[self.layers[j]][:, :-n], size=[ny, nx], mode='bilinear', align_corners=False)

        elif ONNX_EXPORT:
            bs = 1  # batch size
        else:
            bs, _, ny, nx = p.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p

        elif ONNX_EXPORT:
            # Avoid broadcasting for ANE operations
            m = self.na * self.nx * self.ny
            ng = 1. / self.ng.repeat(m, 1)
            grid = self.grid.repeat(1, self.na, 1, 1, 1).view(m, 2)
            anchor_wh = self.anchor_wh.repeat(1, 1, self.nx, self.ny, 1).view(m, 2) * ng

            p = p.view(m, self.no)
            xy = torch.sigmoid(p[:, 0:2]) + grid  # x, y
            wh = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
            p_cls = torch.sigmoid(p[:, 4:5]) if self.nc == 1 else \
                torch.sigmoid(p[:, 5:self.no]) * torch.sigmoid(p[:, 4:5])  # conf
            return p_cls, xy * ng, wh

        else:  # inference
            io = p.sigmoid()
            try:
                io[..., :2] = (io[..., :2] * 2. - 0.5 + self.grid)
            except:
                self.create_grids((nx, ny), p.device)
                io[..., :2] = (io[..., :2] * 2. - 0.5 + self.grid)

            io[..., 2:4] = (io[..., 2:4] * 2) ** 2 * self.anchor_wh
            io[..., :4] *= self.stride
            #io = p.clone()  # inference output
            #io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy
            #io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            #io[..., :4] *= self.stride
            #torch.sigmoid_(io[..., 4:])
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]


class JDELayer(nn.Module):
    def __init__(self, anchors, nc, img_size, stride):
        super(JDELayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.stride = stride  # layer stride
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs (85)
        self.nx, self.ny, self.ng = 0, 0, 0  # initialize number of x, y gridpoints
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)

        if ONNX_EXPORT:
            self.training = False
            self.create_grids((img_size[1] // stride, img_size[0] // stride))  # number x, y grid points

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

    def forward(self, p, out):
        ASFF = False  # https://arxiv.org/abs/1911.09516
        if ASFF:
            raise ValueError('ASFF is not supported')
            # i, n = self.index, self.nl  # index in layers, number of layers
            # p = out[self.layers[i]]
            # bs, _, ny, nx = p.shape  # bs, 255, 13, 13
            # if (self.nx, self.ny) != (nx, ny):
            #     self.create_grids((nx, ny), p.device)

            # # outputs and weights
            # # w = F.softmax(p[:, -n:], 1)  # normalized weights
            # w = torch.sigmoid(p[:, -n:]) * (2 / n)  # sigmoid weights (faster)
            # # w = w / w.sum(1).unsqueeze(1)  # normalize across layer dimension

            # # weighted ASFF sum
            # p = out[self.layers[i]][:, :-n] * w[:, i:i + 1]
            # for j in range(n):
            #     if j != i:
            #         p += w[:, j:j + 1] * \
            #              F.interpolate(out[self.layers[j]][:, :-n], size=[ny, nx], mode='bilinear', align_corners=False)

        elif ONNX_EXPORT:
            bs = 1  # batch size
        else:
            bs, _, ny, nx = p.shape  # bs, 255, 13, 13
            if (self.nx, self.ny) != (nx, ny):
                self.create_grids((nx, ny), p.device)

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p

        elif ONNX_EXPORT:
            # Avoid broadcasting for ANE operations
            m = self.na * self.nx * self.ny
            ng = 1. / self.ng.repeat(m, 1)
            grid = self.grid.repeat(1, self.na, 1, 1, 1).view(m, 2)
            anchor_wh = self.anchor_wh.repeat(1, 1, self.nx, self.ny, 1).view(m, 2) * ng

            p = p.view(m, self.no)
            xy = torch.sigmoid(p[:, 0:2]) + grid  # x, y
            wh = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
            p_cls = torch.sigmoid(p[:, 4:5]) if self.nc == 1 else \
                torch.sigmoid(p[:, 5:self.no]) * torch.sigmoid(p[:, 4:5])  # conf
            return p_cls, xy * ng, wh

        else:  # inference
            #io = p.sigmoid()
            #io[..., :2] = (io[..., :2] * 2. - 0.5 + self.grid)
            #io[..., 2:4] = (io[..., 2:4] * 2) ** 2 * self.anchor_wh
            #io[..., :4] *= self.stride
            io = p.clone()  # inference output
            io[..., :2] = torch.sigmoid(io[..., :2]) * 2. - 0.5 + self.grid  # xy
            io[..., 2:4] = (torch.sigmoid(io[..., 2:4]) * 2) ** 2 * self.anchor_wh  # wh yolo method
            io[..., :4] *= self.stride
            io[..., 4:] = F.softmax(io[..., 4:])
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]

class Fused_Darknets(torch.nn.Module):
    def __init__(self, dict, img_size=(416, 416), verbose=False):
        super(Fused_Darknets, self).__init__()
        self.nclasses = dict['nclasses']
        self.anchors = dict['anchors_g']

        self.fused_backbone = Fused_Backbone(H_att_bc=dict['H_attention_bc'], H_att_ac=dict['H_attention_ac'], spatial=dict['spatial'])
        self.neck = Neck()
        self.head = Head(self.nclasses)
        self.yolo3 = YOLOLayer(self.anchors[0:3], self.nclasses, img_size, stride = 8)
        self.yolo4 = YOLOLayer(self.anchors[3:6], self.nclasses, img_size, stride = 16)
        self.yolo5 = YOLOLayer(self.anchors[6:9], self.nclasses, img_size, stride = 32)
        self.yolo_layers = get_yolo_layers(self)

        # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training
        self.info(verbose) if not ONNX_EXPORT else None  # print model description

    def forward(self, x, y=None, augment=False, verbose=False):

        if y is None:
            y = x[:,-1,:,:].unsqueeze(0).to("cuda")
            x = x[:,:3,:,:].to("cuda")

        if not augment:
            return self.forward_once(x, y)
        else:  # Augment images (inference and test only) https://github.com/ultralytics/yolov3/issues/931
            img_size = x.shape[-2:]  # height, width
            s = [0.83, 0.67]  # scales
            y = []
            for i, xi in enumerate((x,
                                    torch_utils.scale_img(x.flip(3), s[0], same_shape=False),  # flip-lr and scale
                                    torch_utils.scale_img(x, s[1], same_shape=False),  # scale
                                    )):
                # cv2.imwrite('img%g.jpg' % i, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])
                y.append(self.forward_once(xi)[0])

            y[1][..., :4] /= s[0]  # scale
            y[1][..., 0] = img_size[1] - y[1][..., 0]  # flip lr
            y[2][..., :4] /= s[1]  # scale

            y = torch.cat(y, 1)
            return y, None

    def forward_once(self, rgb, ir, augment=False, verbose=False):
    # def forward(self, x, augment=False, verbose=False):
        img_size = rgb.shape[-2:]  # height, width
        # x3_rgb, x4_rgb, x5_rgb = self.backbone(x)
        # x3_ir, x4_ir, x5_ir = self.backbone(y)
        # self.x3_f, self.x4_f, self.x5_f = torch.cat((x3_ir, x3_rgb)), torch.cat((x4_ir, x4_rgb)), torch.cat((x5_ir, x5_rgb))
        # y3,y4,y5 = self.head(self.neck([self.x3_f, self.x4_f, self.x5_f]))

        # a = self.fused_backbone(rgb, ir)
        # print(a[0].shape)

        y3,y4,y5 = self.head(self.neck(self.fused_backbone(rgb, ir)))
        y3 = self.yolo3(y3)
        y4 = self.yolo4(y4)
        y5 = self.yolo5(y5)
        yolo_out = [y3,y4,y5]
        if verbose:
            print('0', rgb.shape)
            str = ''

        # Augment images (inference and test only) ******
        if augment:  # https://github.com/ultralytics/yolov3/issues/931
            nb = rgb.shape[0]  # batch size
            s = [0.83, 0.67]  # scales
            x = torch.cat((rgb,
                           torch_utils.scale_img(rgb.flip(3), s[0]),  # flip-lr and scale
                           torch_utils.scale_img(rgb, s[1]),  # scale
                           ), 0)

        if self.training:  # train
            return yolo_out
        elif ONNX_EXPORT:  # export
            x = [torch.cat(x, 0) for x in zip(*yolo_out)]
            return x[0], torch.cat(x[1:3], 1)  # scores, boxes: 3780x80, 3780x4
        else:  # inference or test
            x, p = zip(*yolo_out)  # inference output, training output
            x = torch.cat(x, 1)  # cat yolo outputs
            if augment:  # de-augment results
                x = torch.split(x, nb, dim=0)
                x[1][..., :4] /= s[0]  # scale
                x[1][..., 0] = img_size[1] - x[1][..., 0]  # flip lr
                x[2][..., :4] /= s[1]  # scale
                x = torch.cat(x, 1)
            return x, p

    def info(self, verbose=False):
        torch_utils.model_info(self, verbose)


class Darknet(torch.nn.Module):
    # YOLOv3 object detection model

    # # org below
    # def __init__(self, cfg, img_size=(416, 416), verbose=False):
    #     super(Darknet, self).__init__()

        # self.module_defs = parse_model_cfg(cfg)
        # self.module_list, self.routs = create_modules(self.module_defs, img_size, cfg)
        # self.yolo_layers = get_yolo_layers(self)
        # # torch_utils.initialize_weights(self)

        # # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        # self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        # self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training
        # self.info(verbose) if not ONNX_EXPORT else None  # print model description

    def __init__(self, dict, img_size=(416, 416), verbose=False):
        super(Darknet, self).__init__()
        self.nclasses = dict['nclasses']
        self.anchors = dict['anchors_g']

        self.backbone = Backbone(dict['mode'], dict['H_attention_bc'], dict['spatial']) # mode: rgb or ir
        self.neck = Neck()
        self.head = Head(self.nclasses)
        self.yolo3 = YOLOLayer(self.anchors[0:3], self.nclasses, img_size, stride = 8)
        self.yolo4 = YOLOLayer(self.anchors[3:6], self.nclasses, img_size, stride = 16)
        self.yolo5 = YOLOLayer(self.anchors[6:9], self.nclasses, img_size, stride = 32)
        self.yolo_layers = get_yolo_layers(self)

        # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training
        self.info(verbose) if not ONNX_EXPORT else None  # print model description

    def forward(self, x, augment=False, verbose=False):

        if not augment:
            return self.forward_once(x)
        else:  # Augment images (inference and test only) https://github.com/ultralytics/yolov3/issues/931
            img_size = x.shape[-2:]  # height, width
            s = [0.83, 0.67]  # scales
            y = []
            for i, xi in enumerate((x,
                                    torch_utils.scale_img(x.flip(3), s[0], same_shape=False),  # flip-lr and scale
                                    torch_utils.scale_img(x, s[1], same_shape=False),  # scale
                                    )):
                # cv2.imwrite('img%g.jpg' % i, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])
                y.append(self.forward_once(xi)[0])

            y[1][..., :4] /= s[0]  # scale
            y[1][..., 0] = img_size[1] - y[1][..., 0]  # flip lr
            y[2][..., :4] /= s[1]  # scale

            # for i, yi in enumerate(y):  # coco small, medium, large = < 32**2 < 96**2 <
            #     area = yi[..., 2:4].prod(2)[:, :, None]
            #     if i == 1:
            #         yi *= (area < 96. ** 2).float()
            #     elif i == 2:
            #         yi *= (area > 32. ** 2).float()
            #     y[i] = yi

            y = torch.cat(y, 1)
            return y, None

    def forward_once(self, x, augment=False, verbose=False):
    # def forward(self, x, augment=False, verbose=False):
        img_size = x.shape[-2:]  # height, width
        y3,y4,y5 = self.head(self.neck(self.backbone(x)))
        y3 = self.yolo3(y3)
        y4 = self.yolo4(y4)
        y5 = self.yolo5(y5)
        yolo_out = [y3,y4,y5]
        if verbose:
            print('0', x.shape)
            str = ''

        # Augment images (inference and test only) ******
        if augment:  # https://github.com/ultralytics/yolov3/issues/931
            nb = x.shape[0]  # batch size
            s = [0.83, 0.67]  # scales
            x = torch.cat((x,
                           torch_utils.scale_img(x.flip(3), s[0]),  # flip-lr and scale
                           torch_utils.scale_img(x, s[1]),  # scale
                           ), 0)

        # # org below
        # for i, module in enumerate(self.module_list):
        #     name = module.__class__.__name__
        #     if name in ['WeightedFeatureFusion', 'FeatureConcat', 'FeatureConcat2', 'FeatureConcat3', 'FeatureConcat_l']:  # sum, concat
        #         if verbose:
        #             l = [i - 1] + module.layers  # layers
        #             sh = [list(x.shape)] + [list(out[i].shape) for i in module.layers]  # shapes
        #             str = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, sh)])
        #         x = module(x, out)  # WeightedFeatureFusion(), FeatureConcat()
        #     elif name == 'YOLOLayer':
        #         yolo_out.append(module(x, out))
        #     else:  # run module directly, i.e. mtype = 'convolutional', 'upsample', 'maxpool', 'batchnorm2d' etc.
        #         x = module(x)

        #     out.append(x if self.routs[i] else [])
            # if verbose:
            #     print('%g/%g %s -' % (i, len(self.module_list), name), list(x.shape), str)
            #     str = ''
        if self.training:  # train
            return yolo_out
        elif ONNX_EXPORT:  # export
            x = [torch.cat(x, 0) for x in zip(*yolo_out)]
            return x[0], torch.cat(x[1:3], 1)  # scores, boxes: 3780x80, 3780x4
        else:  # inference or test
            x, p = zip(*yolo_out)  # inference output, training output
            x = torch.cat(x, 1)  # cat yolo outputs
            if augment:  # de-augment results
                x = torch.split(x, nb, dim=0)
                x[1][..., :4] /= s[0]  # scale
                x[1][..., 0] = img_size[1] - x[1][..., 0]  # flip lr
                x[2][..., :4] /= s[1]  # scale
                x = torch.cat(x, 1)
            return x, p

    def fuse(self):
        # Fuse Conv2d + BatchNorm2d layers throughout model
        print('Fusing layers...')
        fused_list = nn.ModuleList()
        for a in list(self.children())[0]:
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                        # fuse this bn layer with the previous conv2d layer
                        conv = a[i - 1]
                        fused = torch_utils.fuse_conv_and_bn(conv, b)
                        a = nn.Sequential(fused, *list(a.children())[i + 1:])
                        break
            fused_list.append(a)
        self.module_list = fused_list
        self.info() if not ONNX_EXPORT else None  # yolov3-spp reduced from 225 to 152 layers

    def info(self, verbose=False):
        torch_utils.model_info(self, verbose)


def get_yolo_layers(model):
    # return [i for i, m in enumerate(model.module_list) if m.__class__.__name__ == 'YOLOLayer']  # [89, 101, 113]
    return [89, 101, 113]

def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'

    # Establish cutoffs (load layers between 0 and cutoff. if cutoff = -1 all are loaded)
    file = Path(weights).name
    if file == 'darknet53.conv.74':
        cutoff = 75
    elif file == 'yolov3-tiny.conv.15':
        cutoff = 15

    # Read weights file
    with open(weights, 'rb') as f:
        # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.fromfile(f, dtype=np.int32, count=3)  # (int32) version info: major, minor, revision
        self.seen = np.fromfile(f, dtype=np.int64, count=1)  # (int64) number of images seen during training

        weights = np.fromfile(f, dtype=np.float32)  # the rest are weights

    ptr = 0
    for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if mdef['type'] == 'convolutional':
            conv = module[0]
            if mdef['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn = module[1]
                nb = bn.bias.numel()  # number of biases
                # Bias
                bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.bias))
                ptr += nb
                # Weight
                bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.weight))
                ptr += nb
                # Running Mean
                bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_mean))
                ptr += nb
                # Running Var
                bn.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr + nb]).view_as(bn.running_var))
                ptr += nb
            else:
                # Load conv. bias
                nb = conv.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + nb]).view_as(conv.bias)
                conv.bias.data.copy_(conv_b)
                ptr += nb
            # Load conv. weights
            nw = conv.weight.numel()  # number of weights
            conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + nw]).view_as(conv.weight))
            ptr += nw


def save_weights(self, path='model.weights', cutoff=-1):
    # Converts a PyTorch model to Darket format (*.pt to *.weights)
    # Note: Does not work if model.fuse() is applied
    with open(path, 'wb') as f:
        # Write Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version.tofile(f)  # (int32) version info: major, minor, revision
        self.seen.tofile(f)  # (int64) number of images seen during training

        # Iterate through layers
        for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if mdef['type'] == 'convolutional':
                conv_layer = module[0]
                # If batch norm, load bn first
                if mdef['batch_normalize']:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(f)
                    bn_layer.weight.data.cpu().numpy().tofile(f)
                    bn_layer.running_mean.data.cpu().numpy().tofile(f)
                    bn_layer.running_var.data.cpu().numpy().tofile(f)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(f)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(f)


def convert(cfg='cfg/yolov3-spp.cfg', weights='weights/yolov3-spp.weights', saveto='converted.weights'):
    # Converts between PyTorch and Darknet format per extension (i.e. *.weights convert to *.pt and vice versa)
    # from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')

    # Initialize model
    model = Darknet(cfg)
    ckpt = torch.load(weights)  # load checkpoint
    try:
        ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(ckpt['model'], strict=False)
        save_weights(model, path=saveto, cutoff=-1)
    except KeyError as e:
        print(e)

def attempt_download(weights):
    # Attempt to download pretrained weights if not found locally
    weights = weights.strip()
    msg = weights + ' missing, try downloading from https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0'

    if len(weights) > 0 and not os.path.isfile(weights):
        d = {''}

        file = Path(weights).name
        if file in d:
            r = gdrive_download(id=d[file], name=weights)
        else:  # download from pjreddie.com
            url = 'https://pjreddie.com/media/files/' + file
            print('Downloading ' + url)
            r = os.system('curl -f ' + url + ' -o ' + weights)

        # Error check
        if not (r == 0 and os.path.exists(weights) and os.path.getsize(weights) > 1E6):  # weights exist and > 1MB
            os.system('rm ' + weights)  # remove partial downloads
            raise Exception(msg)
