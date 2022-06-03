import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class convbnrelu(nn.Sequential):
    def __init__(self, inp, oup, ker=3, stride=1, groups=1):
        super(convbnrelu, self).__init__(
            nn.Conv2d(inp, oup, ker, stride, ker // 2, groups=groups, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )

class Bottleneck(nn.Module):

    def __init__(self, inp, oup, s=1, k=3, r=4):
        super(Bottleneck, self).__init__()
        mid_dim = oup // r
        if inp == oup and s == 1:
            self.residual = True
        else:
            self.residual = False
        self.conv1 = nn.Conv2d(inp, mid_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_dim)
        self.conv2 = nn.Conv2d(mid_dim, mid_dim, kernel_size=k, stride=s, padding=k//2, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_dim)
        self.conv3 = nn.Conv2d(mid_dim, oup, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(oup)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.residual == True:
            out += residual
        out = self.relu(out)
        return out

class UpConv(nn.Module):
    def __init__(self, inp, oup, k=3):
        super(UpConv, self).__init__()
        self.conv = nn.Conv2d(inp, oup, k, 1, k // 2, bias=False)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2)
        x = self.conv(x)
        return x

class FusedMBConv(nn.Module):

    def __init__(self, inp, oup, s=1, k=3, r=4):
        super(FusedMBConv, self).__init__()
        feature_dim = _make_divisible(round(inp * r), 8)
        self.inv = nn.Sequential(
            nn.Conv2d(inp, feature_dim, k, s, k // 2, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU6(inplace = True)
        )
        self.point_conv = nn.Sequential(
            nn.Conv2d(feature_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        )
        self.use_residual_connection = s == 1 and inp == oup
        
    def forward(self, x):
        out = self.inv(x)
        out = self.point_conv(out)
        if self.use_residual_connection:
            out += x
        return out

class InvBottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, ker=3, exp=6):
        super(InvBottleneck, self).__init__()
        feature_dim = _make_divisible(round(inplanes * exp), 8)
        self.inv = nn.Sequential(
            nn.Conv2d(inplanes, feature_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU6(inplace = True)
        )
        self.depth_conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, ker, stride, ker // 2, groups=feature_dim, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU6(inplace = True)
        )
        self.point_conv = nn.Sequential(
            nn.Conv2d(feature_dim, planes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.stride = stride
        self.use_residual_connection = stride == 1 and inplanes == planes
        
    def forward(self, x):
        out = self.inv(x)
        out = self.depth_conv(out)
        out = self.point_conv(out)
        if self.use_residual_connection:
            out += x
        return out

class SepConv2d(nn.Module):
    def __init__(self, inp, oup, ker=3, stride=1):
        super(SepConv2d, self).__init__()
        conv = [
            nn.Conv2d(inp, inp, ker, stride, ker // 2, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        ]
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        output = self.conv(x)
        return output