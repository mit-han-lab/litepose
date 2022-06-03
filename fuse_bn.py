# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import json
import copy

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import time
import torch.utils.data.distributed
import torchvision.transforms
import torch.multiprocessing
from tqdm import tqdm

import _init_paths
import models

from config import cfg
from config import check_config
from config import update_config
from core.inference import get_multi_stage_outputs
from core.inference import aggregate_results
from core.group import HeatmapParser
from dataset import make_test_dataloader, make_train_dataloader
from fp16_utils.fp16util import network_to_half
from utils.utils import create_logger
from utils.utils import get_model_summary
from utils.vis import save_debug_images
from utils.vis import save_valid_image
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size
from arch_manager import ArchManager

torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
                        
    #fixed config for supernet
    parser.add_argument('--superconfig',
                        default=None,
                        type=str,
                        help='fixed arch for supernet training')

    args = parser.parse_args()

    return args
    
def rm_bn_from_net(net):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.forward = lambda x: x

def fuse_conv_and_bn(conv, bn):
    with torch.no_grad():
        # init
        fusedconv = torch.nn.Conv2d(conv.in_channels,
                                    conv.out_channels,
                                    kernel_size=conv.kernel_size,
                                    stride=conv.stride,
                                    padding=conv.padding,
                                    groups=conv.groups,
                                    bias=True)

        # prepare filters
        bn_weight = bn.weight.clone()
        bn_weight.div_(torch.sqrt(bn.eps + bn.running_var))
        fusedconv.weight.copy_(bn_weight.view(conv.out_channels, 1, 1, 1) * conv.weight)

        # prepare spatial bias
        if conv.bias is not None:
            b_conv = conv.bias
        else:
            b_conv = torch.zeros(conv.weight.size(0))

        b_conv = bn_weight * b_conv.view(-1, conv.out_channels).view(fusedconv.bias.size())

        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(b_conv + b_bn)

        return fusedconv

def fuse_deconv_and_bn(deconv, bn, op=1):
    with torch.no_grad():
        # init
        FusedDeconv = torch.nn.ConvTranspose2d(deconv.in_channels,
                                             deconv.out_channels,
                                             kernel_size=deconv.kernel_size,
                                             stride=deconv.stride,
                                             padding=deconv.padding,
                                             bias=True
                                             )

        # prepare filters
        bn_weight=bn.weight.clone()
        bn_weight.div_(torch.sqrt(bn.eps + bn.running_var))
        FusedDeconv.weight.copy_(bn_weight.view(1, deconv.out_channels, 1, 1)*deconv.weight)

        # prepare spatial bias
        if deconv.bias is not None:
            b_conv = deconv.bias
        else:
            b_conv = torch.zeros(deconv.weight.size(1))

        b_conv = bn_weight*b_conv.view(-1,deconv.out_channels).view(FusedDeconv.bias.size())

        b_bn = op * (bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps)))
        FusedDeconv.bias.copy_(b_conv + b_bn)

        return FusedDeconv

def fuse_cbr(cbr):
    cbr[0] = fuse_conv_and_bn(cbr[0], cbr[1])

def fuse_inv(inv):
    fuse_cbr(inv.inv)
    fuse_cbr(inv.depth_conv)
    fuse_cbr(inv.point_conv)

def transfer(net, arch):
    fuse_cbr(net.first[0])
    fuse_cbr(net.first[1])
    net.first[2] = fuse_conv_and_bn(net.first[2], net.first[3])
    backbone_setting = arch['backbone_setting']
    for id_stage in range(len(backbone_setting)):
        n = backbone_setting[id_stage]['num_blocks']
        for id_block in range(n):
            fuse_inv(net.stage[id_stage][id_block])
    num_deconv_layers = 3
    for i in range(num_deconv_layers):
        net.deconv_refined[i] = fuse_deconv_and_bn(net.deconv_refined[i], net.deconv_bnrelu[i][0])
        net.deconv_raw[i] = fuse_deconv_and_bn(net.deconv_raw[i], net.deconv_bnrelu[i][0], op=0)
        if i > 0:
            fuse_cbr(net.final_refined[i-1].conv)
            fuse_cbr(net.final_raw[i-1].conv)

def diff(a, b):
    print(((a-b)**2).mean())

def abssum(a):
    print((torch.abs(a)).sum())

def main():
    args = parse_args()
    update_config(cfg, args)
    check_config(cfg)

    # change the resolution according to config
    fixed_arch = None
    with open(args.superconfig, 'r') as f:
        fixed_arch = json.load(f)
    cfg.defrost()
    reso = fixed_arch['img_size']
    cfg.DATASET.INPUT_SIZE = reso
    cfg.DATASET.OUTPUT_SIZE = [reso // 4, reso // 2]
    cfg.TEST.FLIP_TEST = False
    cfg.TEST.ADJUST = False
    cfg.TEST.REFINE = False
    cfg.freeze()

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    arch_manager = ArchManager(cfg)
    cfg_arch = fixed_arch
    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True, cfg_arch = cfg_arch
    )
    # set eval mode
    model.eval()
    dump_input = torch.ones(
        (1, 3, cfg.DATASET.INPUT_SIZE, cfg.DATASET.INPUT_SIZE)
    )
    print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
    model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
    model_ori = copy.deepcopy(model)
    transfer(model, fixed_arch)
    rm_bn_from_net(model)
    with torch.no_grad():
        o1 = model_ori(dump_input)
        o2 = model(dump_input)

    diff(o1[0], o2[0])
    abssum(o1[0])
    abssum(o1[1])
    abssum(o2[0])
    abssum(o2[1])
    torch.save(model.state_dict(), './output/crowd_pose_kpt/pose_mobilenet/mobile/model_fuse_bn.pth.tar')


if __name__ == '__main__':
    main()
