from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import logging
import torch
import torch.nn as nn
import math
from models.layers.efficient_blocks import conv_bn_act, SamePadConv2d, Flatten, SEModule, DropConnect, conv, conv_dw_no_bn,conv, conv_bn, conv_pw

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

#EfficientNet modules
class Swish(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Swish, self).__init__()
    def forward(self, x):
        return x * torch.sigmoid(x)

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        padding = self._get_padding(kernel_size, stride)
        super(ConvBNReLU, self).__init__(
            nn.ZeroPad2d(padding),
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding=0, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            Swish(),
        )
    def _get_padding(self, kernel_size, stride):
        p = max(kernel_size - stride, 0)
        return [p // 2, p - p // 2, p // 2, p - p // 2]


class SqueezeExcitation(nn.Module):
    def __init__(self, in_planes, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, reduced_dim, 1),
            Swish(),
            nn.Conv2d(reduced_dim, in_planes, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.se(x)

class MBConvBlock(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 expand_ratio,
                 kernel_size,
                 stride,
                 reduction_ratio=4,
                 drop_connect_rate=0.2):
        super(MBConvBlock, self).__init__()
        self.drop_connect_rate = drop_connect_rate
        self.use_residual = in_planes == out_planes and stride == 1
        assert stride in [1, 2]
        assert kernel_size in [3, 5]

        hidden_dim = in_planes * expand_ratio
        reduced_dim = max(1, int(in_planes / reduction_ratio))

        layers = []
        # pw
        if in_planes != hidden_dim:
            layers += [ConvBNReLU(in_planes, hidden_dim, 1)]

        layers += [
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, kernel_size, stride=stride, groups=hidden_dim),
            # se
            SqueezeExcitation(hidden_dim, reduced_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, out_planes, 1, bias=False),
            nn.BatchNorm2d(out_planes),
        ]

        self.conv = nn.Sequential(*layers)

    def _drop_connect(self, x):
        if not self.training:
            return x
        keep_prob = 1.0 - self.drop_connect_rate
        batch_size = x.size(0)
        random_tensor = keep_prob
        random_tensor += torch.rand(batch_size, 1, 1, 1, device=x.device)
        binary_tensor = random_tensor.floor()
        return x.div(keep_prob) * binary_tensor

    def forward(self, x):
        if self.use_residual:
            return x + self._drop_connect(self.conv(x))
        else:
            return self.conv(x)

def _make_divisible(value, divisor=8):
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor 
    return new_value

def _round_filters(filters, width_mult):
    if width_mult == 1.0:
        return filters
    return int(_make_divisible(filters * width_mult))

def _round_repeats(repeats, depth_mult):
    if depth_mult == 1.0:
        return repeats
    return int(math.ceil(depth_mult * repeats))

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                               momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        nn.BatchNorm2d(num_inchannels[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3),
                                nn.ReLU(True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse
blocks_dict = {
    'BASIC': BasicBlock,
    #'BOTTLENECK': Bottleneck
}
class PoseHigherResolutionNet(nn.Module):
    def __init__(self,cfg, **kwargs):
        super(PoseHigherResolutionNet, self).__init__()
        self.depth_mult = cfg.MODEL.DEPTH_MULT
        settings = [
            # t,  c, n, s, k
            [1,  16, 1, 1, 3],  # MBConv1_3x3, SE, 184
            [6,  24, 2, 2, 3],  # MBConv6_3x3, SE, 92
            [6,  40, 2, 2, 5],  # MBConv6_5x5, SE, 46
            [6,  80, 3, 2, 3],  # MBConv6_3x3, SE, 46
            [6, 112, 3, 1, 5],  # MBConv6_5x5, SE, 46
            [6, 192, 4, 2, 5],  # MBConv6_5x5, SE, 46
            [6, 320, 1, 1, 3]   # MBConv6_3x3, SE, 46
        ]
        # yapf: enable
        out_channels = _round_filters(32, cfg.MODEL.WIDTH_MULT)#width_mult)
        features = [ConvBNReLU(3, out_channels, 3, stride=2)]

        in_channels = out_channels
        for t, c, n, s, k in settings:
            out_channels = _round_filters(c, cfg.MODEL.WIDTH_MULT)#width_mult)
            repeats = _round_repeats(n, cfg.MODEL.DEPTH_MULT)#self.depth_mult)

            for i in range(repeats):
                stride = s if i == 0 else 1
                features += [MBConvBlock(in_channels, out_channels, expand_ratio=t, stride=stride, kernel_size=k)]
                in_channels = out_channels
        self.features = nn.Sequential(*features)
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / math.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA

        self.stage2_cfg = cfg['MODEL']['EXTRA']['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            int(math.ceil(num_channels[i] * (pow(1.2455,cfg.MODEL.SCALE_FACTOR)) * block.expansion))  for i in range(len(num_channels))
        ]
        if cfg.MODEL.WIDTH_MULT == 1 and cfg.MODEL.DEPTH_MULT == 1:
            self.trans1_branch1 = nn.Sequential(nn.Conv2d(24,32,3,1,1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
            self.trans1_branch2 = nn.Sequential(nn.Conv2d(40,64,3,1,1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)) 
        else:               
            self.trans1_branch1 = nn.Sequential(nn.Conv2d(int(_make_divisible(24*cfg.MODEL.WIDTH_MULT)),int(math.ceil(32*pow(1.2455,cfg.MODEL.SCALE_FACTOR))),3,1,1), 
                                                nn.BatchNorm2d(int(math.ceil(32*pow(1.2455,cfg.MODEL.SCALE_FACTOR)))), nn.ReLU(inplace=True))
            self.trans1_branch2 = nn.Sequential(nn.Conv2d(int(_make_divisible(40*cfg.MODEL.WIDTH_MULT)),int(math.ceil(64*pow(1.2455,cfg.MODEL.SCALE_FACTOR))),3,1,1), 
                                                nn.BatchNorm2d(int(math.ceil(64*pow(1.2455,cfg.MODEL.SCALE_FACTOR)))), nn.ReLU(inplace=True))

        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels, cfg) #should happen automatically

        self.stage3_cfg = cfg['MODEL']['EXTRA']['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            int(math.ceil(num_channels[i] * (pow(1.2455,cfg.MODEL.SCALE_FACTOR)) * block.expansion))  for i in range(len(num_channels))
        ]
        if cfg.MODEL.WIDTH_MULT == 1 and cfg.MODEL.DEPTH_MULT == 1:
            self.trans2_branch1 = nn.Sequential(nn.Conv2d(32,32,3,1,1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
            self.trans2_branch2 = nn.Sequential(nn.Conv2d(64,64,3,1,1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
            self.trans2_branch3 = nn.Sequential(nn.Conv2d(112,128,3,1,1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        else:
            self.trans2_branch1 = nn.Sequential(nn.Conv2d(int(math.ceil(32*pow(1.2455,cfg.MODEL.SCALE_FACTOR))), int(math.ceil(32*pow(1.2455,cfg.MODEL.SCALE_FACTOR))),3,1,1),
                                                nn.BatchNorm2d(int(math.ceil(32*pow(1.2455,cfg.MODEL.SCALE_FACTOR)))), nn.ReLU(inplace=True))
            self.trans2_branch2 = nn.Sequential(nn.Conv2d(int(math.ceil(64*pow(1.2455,cfg.MODEL.SCALE_FACTOR))), int(math.ceil(64*pow(1.2455,cfg.MODEL.SCALE_FACTOR))),3,1,1),
                                                nn.BatchNorm2d(int(math.ceil(64*pow(1.2455,cfg.MODEL.SCALE_FACTOR)))), nn.ReLU(inplace=True))
            self.trans2_branch3 = nn.Sequential(nn.Conv2d(int(_make_divisible(112*cfg.MODEL.WIDTH_MULT)), int(math.ceil(128*pow(1.2455,cfg.MODEL.SCALE_FACTOR))),3,1,1),
                                                nn.BatchNorm2d(int(math.ceil(128*pow(1.2455,cfg.MODEL.SCALE_FACTOR)))), nn.ReLU(inplace=True))

        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels, cfg) #should happen automatically

        self.stage4_cfg = cfg['MODEL']['EXTRA']['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            int(math.ceil(num_channels[i] * (pow(1.2455,cfg.MODEL.SCALE_FACTOR)) * block.expansion))  for i in range(len(num_channels))
        ]
        if cfg.MODEL.WIDTH_MULT == 1 and cfg.MODEL.DEPTH_MULT == 1:
            self.trans3_branch1 = nn.Sequential(nn.Conv2d(32,32,3,1,1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
            self.trans3_branch2 = nn.Sequential(nn.Conv2d(64,64,3,1,1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
            self.trans3_branch3 = nn.Sequential(nn.Conv2d(128,128,3,1,1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
            self.trans3_branch4 = nn.Sequential(nn.Conv2d(320,256,3,1,1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        else:
            self.trans3_branch1 = nn.Sequential(nn.Conv2d(int(math.ceil(32*pow(1.2455,cfg.MODEL.SCALE_FACTOR))), int(math.ceil(32*pow(1.2455,cfg.MODEL.SCALE_FACTOR))),3,1,1),
                                                nn.BatchNorm2d(int(math.ceil(32*pow(1.2455,cfg.MODEL.SCALE_FACTOR)))), nn.ReLU(inplace=True))
            self.trans3_branch2 = nn.Sequential(nn.Conv2d(int(math.ceil(64*pow(1.2455,cfg.MODEL.SCALE_FACTOR))), int(math.ceil(64*pow(1.2455,cfg.MODEL.SCALE_FACTOR))),3,1,1),
                                                nn.BatchNorm2d(int(math.ceil(64*pow(1.2455,cfg.MODEL.SCALE_FACTOR)))), nn.ReLU(inplace=True))
            self.trans3_branch3 = nn.Sequential(nn.Conv2d(int(math.ceil(128*pow(1.2455,cfg.MODEL.SCALE_FACTOR))), int(math.ceil(128*pow(1.2455,cfg.MODEL.SCALE_FACTOR))),3,1,1),
                                                nn.BatchNorm2d(int(math.ceil(128*pow(1.2455,cfg.MODEL.SCALE_FACTOR)))), nn.ReLU(inplace=True))
            self.trans3_branch4 = nn.Sequential(nn.Conv2d(int(_make_divisible(320*cfg.MODEL.WIDTH_MULT)), int(math.ceil(256*pow(1.2455,cfg.MODEL.SCALE_FACTOR))),3,1,1),
                                                nn.BatchNorm2d(int(math.ceil(256*pow(1.2455,cfg.MODEL.SCALE_FACTOR)))), nn.ReLU(inplace=True))
        
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, cfg, multi_scale_output=False) #should happen automatically

        self.final_layers = self._make_final_layers(cfg, pre_stage_channels[0])
        self.deconv_layers = self._make_deconv_layers(
            cfg, pre_stage_channels[0])

        self.num_deconvs = extra.DECONV.NUM_DECONVS
        self.deconv_config = cfg.MODEL.EXTRA.DECONV
        self.loss_config = cfg.LOSS

        self.pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']

    def _make_final_layers(self, cfg, input_channels):
        dim_tag = cfg.MODEL.NUM_JOINTS if cfg.MODEL.TAG_PER_JOINT else 1
        extra = cfg.MODEL.EXTRA
        #scale_factor = 0
        final_layers = []
        output_channels = cfg.MODEL.NUM_JOINTS + dim_tag \
            if cfg.LOSS.WITH_AE_LOSS[0] else cfg.MODEL.NUM_JOINTS
        final_layers.append(nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        ))

        deconv_cfg = extra.DECONV
        for i in range(deconv_cfg.NUM_DECONVS):
            input_channels = int(math.ceil(deconv_cfg.NUM_CHANNELS[i]*pow(1.2455, cfg.MODEL.SCALE_FACTOR)))
            output_channels = cfg.MODEL.NUM_JOINTS + dim_tag \
                if cfg.LOSS.WITH_AE_LOSS[i+1] else cfg.MODEL.NUM_JOINTS
            final_layers.append(nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
            ))

        return nn.ModuleList(final_layers)

    def _make_deconv_layers(self, cfg, input_channels):
        dim_tag = cfg.MODEL.NUM_JOINTS if cfg.MODEL.TAG_PER_JOINT else 1
        extra = cfg.MODEL.EXTRA
        deconv_cfg = extra.DECONV
        #scale_factor = 0
        deconv_layers = []
        for i in range(deconv_cfg.NUM_DECONVS):
            if deconv_cfg.CAT_OUTPUT[i]:
                final_output_channels = cfg.MODEL.NUM_JOINTS + dim_tag \
                    if cfg.LOSS.WITH_AE_LOSS[i] else cfg.MODEL.NUM_JOINTS
                input_channels += final_output_channels
            output_channels = int(math.ceil(deconv_cfg.NUM_CHANNELS[i]*pow(1.2455, cfg.MODEL.SCALE_FACTOR)))
            deconv_kernel, padding, output_padding = \
                self._get_deconv_cfg(deconv_cfg.KERNEL_SIZE[i])

            layers = []
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=deconv_kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False),
                nn.BatchNorm2d(output_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            ))
            for _ in range(cfg.MODEL.EXTRA.DECONV.NUM_BASIC_BLOCKS):
                layers.append(nn.Sequential(
                    BasicBlock(output_channels, output_channels),
                ))
            deconv_layers.append(nn.Sequential(*layers))
            input_channels = output_channels

        return nn.ModuleList(deconv_layers)

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, cfg,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        #scale_factor = 0
        num_channels = [
            int(math.ceil(num_channels[i] * (pow(1.2455,cfg.MODEL.SCALE_FACTOR))))  for i in range(len(num_channels))
        ]
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        if self.depth_mult == 0.483: #bc4
            #print("HERE!!!!")
            for i in range(0,3):
                x = self.features[i](x) #96x96
            x1 = x
            for i in range(3,4): 
                x = self.features[i](x) #48x48
            x2 = x
            for i in range(4,8):
                x = self.features[i](x) #24x24
            x3 = x
            for i in range(8,11):
                x = self.features[i](x) #12x12
            x4 = x
        if self.depth_mult == 0.578: #bc3
            for i in range(0,4):
                x = self.features[i](x) #104x104
            x1 = x
            for i in range(4,6):
                x = self.features[i](x) #52x52
            x2 = x
            for i in range(6,10):
                x = self.features[i](x) #26x26
            x3 = x
            for i in range(10,14):
                x = self.features[i](x) #13x13
            x4 = x
        if self.depth_mult == 0.694: #bc2
            for i in range(0,4):
                x = self.features[i](x) #112x112
            x1 = x
            for i in range(4,6):
                x = self.features[i](x) #56x56
            x2 = x
            for i in range(6,12):
                x = self.features[i](x) #28x28
            x3 = x
            for i in range(12,16):
                x = self.features[i](x) #14x14
            x4 = x
        if self.depth_mult == 1 or self.depth_mult == 0.833: #b0
            for i in range(0,4):
                x = self.features[i](x) #128x128, 120x120
            x1 = x
            for i in range(4,6):
                x = self.features[i](x) #64x64, 60x60
            x2 = x
            for i in range(6,12):
                x = self.features[i](x) #32x32, 30x30
            x3 = x
            for i in range(12,17):
                x = self.features[i](x) #16x16, 15x15
            x4 = x
        elif self.depth_mult == 1.1 or self.depth_mult == 1.2: #b1 and b2
            for i in range(0,6):
                x = self.features[i](x) 
            x1 = x
            for i in range(6,9):
                x = self.features[i](x) 
            x2 = x
            for i in range(9,17):
                x = self.features[i](x)
            x3 = x
            for i in range(17,24):
                x = self.features[i](x) 
            x4 = x
        elif self.depth_mult== 1.4: #b3
            for i in range(0,6):
                x = self.features[i](x) 
            x1 = x
            for i in range(6,9):
                x = self.features[i](x)
            x2 = x
            for i in range(9,19):
                x = self.features[i](x) 
            x3 = x
            for i in range(19,27):
                x = self.features[i](x) 
            x4 = x
        elif self.depth_mult == 1.8: #b4
            for i in range(0,7):
                x = self.features[i](x) 
            x1 = x
            for i in range(7,11):
                x = self.features[i](x) 
            x2 = x
            for i in range(11,23):
                x = self.features[i](x) 
            x3 = x
            for i in range(23,33):
                x = self.features[i](x) 
            x4 = x
        '''
        else: #b5
            for i in range(0,9):
                x = self.features[i](x)
            x1 = x
            for i in range(9,14):
                x = self.features[i](x) 
            x2 = x
            for i in range(14,28):
                x = self.features[i](x)
            x3 = x
            for i in range(28,40):
                x = self.features[i](x) 
            x4 = x
        '''
        x_list = []
        x_list.append(self.trans1_branch1(x1))
        x_list.append(self.trans1_branch2(x2))
        
        y_list = self.stage2(x_list)

        x_list = []
        x_list.append(self.trans2_branch1(y_list[-2]))
        x_list.append(self.trans2_branch2(y_list[-1]))
        x_list.append(self.trans2_branch3(x3))
        
        y_list = self.stage3(x_list)

        x_list = []
        x_list.append(self.trans3_branch1(y_list[-3]))
        x_list.append(self.trans3_branch2(y_list[-2]))
        x_list.append(self.trans3_branch3(y_list[-1]))
        x_list.append(self.trans3_branch4(x4))
        
        y_list = self.stage4(x_list)

        final_outputs = []
        x = y_list[0]
        y = self.final_layers[0](x)
        final_outputs.append(y)

        for i in range(self.num_deconvs):
            if self.deconv_config.CAT_OUTPUT[i]:
                x = torch.cat((x, y), 1)

            x = self.deconv_layers[i](x)
            y = self.final_layers[i+1](x)
            final_outputs.append(y)

        return final_outputs

    def init_weights(self, pretrained='', verbose=True):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        parameters_names = set()
        for name, _ in self.named_parameters():
            parameters_names.add(name)

        buffers_names = set()
        for name, _ in self.named_buffers():
            buffers_names.add(name)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] is '*':
                    if name in parameters_names or name in buffers_names:
                        if verbose:
                            logger.info(
                                '=> init {} from {}'.format(name, pretrained)
                            )
                        need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)

def get_pose_net(cfg, is_train, **kwargs):
    model = PoseHigherResolutionNet(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED, verbose=cfg.VERBOSE)

    return model