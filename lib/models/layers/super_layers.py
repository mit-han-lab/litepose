import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

BN_MOMENTUM = 0.1

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class SuperBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(SuperBatchNorm2d, self).__init__(num_features, eps, momentum, affine)
    def forward(self, x):
        input = x
        in_c = x.shape[1]
        ret = F.batch_norm(
            input, self.running_mean[:in_c], self.running_var[:in_c], self.weight[:in_c], self.bias[:in_c],
            self.training, self.momentum, self.eps)
        return ret[:, :x.shape[1]]

class SuperConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros'):
        super(SuperConvTranspose2d, self).__init__(in_channels, out_channels,
                                                   kernel_size, stride, padding,
                                                   output_padding, groups, bias,
                                                   dilation, padding_mode)

    def forward(self, x, num_channel, output_size=None):
        output_padding = self._output_padding(x, output_size, self.stride, self.padding, self.kernel_size)
        in_nc = x.size(1) 
        out_nc = num_channel
        weight = self.weight[:in_nc, :out_nc]  # [ic, oc, H, W]
        if self.bias is not None:
            bias = self.bias[:out_nc]
        else:
            bias = None
        return F.conv_transpose2d(x, weight, bias, self.stride, self.padding,
                                  output_padding, self.groups, self.dilation)


# No Residual Connection
class SuperSepConv2d(nn.Module):
    def __init__(self, inp, oup, ker=3, stride=1):
        super(SuperSepConv2d, self).__init__()
        conv = [
            nn.Conv2d(inp, inp, ker, stride, ker // 2, groups=inp, bias=False),
            SuperBatchNorm2d(inp),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        ]
        self.conv = nn.Sequential(*conv)

    def forward(self, x, num_channel):
        out_nc = num_channel
        in_nc = x.size(1)
        conv = self.conv[0]
        weight = conv.weight[:in_nc]  # [oc, 1, H, W]
        if conv.bias is not None:
            bias = conv.bias[:in_nc]
        else:
            bias = None
        x = F.conv2d(x, weight, bias, conv.stride, conv.padding, conv.dilation, in_nc)
        x = self.conv[1](x)
        x = self.conv[2](x)
        conv = self.conv[3]
        weight = conv.weight[:out_nc, :in_nc]  # [oc, ic, H, W]
        if conv.bias is not None:
            bias = conv.bias[:out_nc]
        else:
            bias = None
        x = F.conv2d(x, weight, bias, conv.stride, conv.padding, conv.dilation, conv.groups)
        return x

class SuperConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(SuperConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias, padding_mode)

    def forward(self, x, num_channel):
        in_nc = x.size(1)
        out_nc = num_channel
        weight = self.weight[:out_nc, :in_nc]  # [oc, ic, H, W]
        if self.bias is not None:
            bias = self.bias[:out_nc]
        else:
            bias = None
        return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)

class SuperUpConv(nn.Module):
    def __init__(self, inp, oup, k=3):
        super(SuperUpConv, self).__init__()
        self.conv = SuperConv2d(inp, oup, k, 1, k // 2, bias=False)
    def forward(self, x, out_nc):
        x = F.interpolate(x, scale_factor=2)
        x = self.conv(x, out_nc)
        return x

class SuperFusedMBConv(nn.Module):

    def __init__(self, inp, oup, s=1, k=3, r=4):
        super(SuperFusedMBConv, self).__init__()
        feature_dim = _make_divisible(round(inp * r), 8)
        self.inv = nn.Sequential(
            SuperConv2d(inp, feature_dim, k, s, k // 2, bias=False),
            SuperBatchNorm2d(feature_dim),
            nn.ReLU6(inplace = True)
        )
        self.point_conv = nn.Sequential(
            SuperConv2d(feature_dim, oup, 1, 1, 0, bias=False),
            SuperBatchNorm2d(oup)
        )
        self.use_residual_connection = s == 1 and inp == oup
        
    def forward(self, x, out_nc, expansion):
        residual = x
        mid_dim = round(int(x.size(1)) * expansion)
        out = self.inv[0](x, mid_dim)
        out = self.inv[1](out)
        out = self.inv[2](out)
        out = self.point_conv[0](out, out_nc)
        out = self.point_conv[1](out)
        if self.use_residual_connection:
            out = out + residual
        return out

class SuperInvBottleneck(nn.Module):
    # 6 is the upperbound for expansion
    # 7 is the upperbound for kernel_size
    expansion = 6
    ker_size = 7
    def __init__(self, inplanes, planes, stride=1):
        super(SuperInvBottleneck, self).__init__()
        feature_dim = round(inplanes * self.expansion)
        self.inv = nn.Sequential(
            SuperConv2d(inplanes, feature_dim, 1, 1, 0, bias=False),
            SuperBatchNorm2d(feature_dim),
            nn.ReLU6(inplace = True)
        )
        self.depth_conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, self.ker_size, stride, self.ker_size // 2, groups=feature_dim, bias=False),
            SuperBatchNorm2d(feature_dim),
            nn.ReLU6(inplace = True)
        )
        self.point_conv = nn.Sequential(
            SuperConv2d(feature_dim, planes, 1, 1, 0, bias=False),
            SuperBatchNorm2d(planes)
        )
        self.stride = stride
        self.Linear5x5 = nn.Linear(25, 25)
        self.Linear3x3 = nn.Linear(9, 9)
        self.use_residual_connection = stride == 1 and inplanes == planes

    def forward(self, x, out_nc, ker_size, expansion):
        residual = x
        mid_dim = round(int(x.size(1)) * expansion)
        out = self.inv[0](x, mid_dim)
        out = self.inv[1](out)
        out = self.inv[2](out)
        conv = self.depth_conv[0]
        l = self.ker_size // 2 - ker_size // 2
        r = self.ker_size // 2 + ker_size // 2 + 1
        weight = conv.weight[:mid_dim, :, l:r, l:r]  # [oc, 1, H, W]
        s0, s1 = weight.shape[0], weight.shape[1]
        if ker_size == 5:
            weight = self.Linear5x5(weight.reshape(s0, s1, -1)).reshape(s0, s1, 5, 5)
        elif ker_size == 3:
            weight = self.Linear3x3(weight.reshape(s0, s1, -1)).reshape(s0, s1, 3, 3)
        if conv.bias is not None:
            bias = conv.bias[:mid_dim]
        else:
            bias = None
        out = F.conv2d(out, weight, bias, conv.stride, ker_size // 2, conv.dilation, mid_dim)
        out = self.depth_conv[1](out)
        out = self.depth_conv[2](out)
        out = self.point_conv[0](out, out_nc)
        out = self.point_conv[1](out)
        if self.use_residual_connection:
            out = out + residual
        return out