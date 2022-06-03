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

def transfer_conv(superconv, conv):
    out_nc = conv.weight.shape[0]
    in_nc = conv.weight.shape[1]
    conv.weight.data = superconv.weight[:out_nc, :in_nc].data.float().clone()
    if conv.bias is not None:
        conv.bias.data = superconv.bias[:out_nc].data.float().clone()

def transfer_deconv(superdeconv, deconv):
    in_nc = deconv.weight.shape[0]
    out_nc = deconv.weight.shape[1]
    deconv.weight.data = superdeconv.weight[:in_nc, :out_nc].data.float().clone()
    if deconv.bias is not None:
        deconv.bias.data = superdeconv.bias[:out_nc].data.float().clone()

def transfer_dwconv(superdwconv, dwconv):
    mid_dim = dwconv.weight.shape[0]
    dwconv.weight.data = superdwconv.weight[:mid_dim].data.float().clone()
    if dwconv.bias is not None:
        dwconv.bias.data = superdwconv.bias[:mid_dim].data.float().clone()

def transfer_bn(superbn, bn):
    # print(superbn.weight.shape)
    # print(superbn.running_mean.shape)
    # print(superbn.running_var.shape)
    # print(superbn.bias.shape)
    in_nc = bn.num_features
    bn.weight.data = superbn.weight[:in_nc].data.float().clone()
    bn.bias.data = superbn.bias[:in_nc].data.float().clone()
    bn.running_mean.data = superbn.running_mean[:in_nc].data.float().clone()
    bn.running_var.data = superbn.running_var[:in_nc].data.float().clone()

def transfer_inv(superinv, inv):
    # inv
    transfer_conv(superinv.inv[0], inv.inv[0])
    transfer_bn(superinv.inv[1], inv.inv[1])
    # depth
    transfer_dwconv(superinv.depth_conv[0], inv.depth_conv[0])
    transfer_bn(superinv.depth_conv[1], inv.depth_conv[1])
    # point
    transfer_conv(superinv.point_conv[0], inv.point_conv[0])
    transfer_bn(superinv.point_conv[1], inv.point_conv[1])

def transfer_sep(supersep, sep):
    transfer_dwconv(supersep.conv[0], sep.conv[0])
    transfer_bn(supersep.conv[1], sep.conv[1])
    transfer_conv(supersep.conv[3], sep.conv[3])

def transfer_cbr(supercbr, cbr):
    transfer_conv(supercbr[0], cbr[0])
    transfer_bn(supercbr[1], cbr[1])

def transfer(supernet, net, arch):
    transfer_cbr(supernet.first[0], net.first[0])
    transfer_cbr(supernet.first[1], net.first[1])
    transfer_conv(supernet.first[2], net.first[2])
    transfer_bn(supernet.first[3], net.first[3])
    backbone_setting = arch['backbone_setting']
    for id_stage in range(len(backbone_setting)):
        n = backbone_setting[id_stage]['num_blocks']
        for id_block in range(n):
            transfer_inv(supernet.stage[id_stage][id_block], net.stage[id_stage][id_block])
    num_deconv_layers = cfg.MODEL.EXTRA.NUM_DECONV_LAYERS
    for i in range(len(net.deconv_refined)):
        transfer_deconv(supernet.deconv_refined[i], net.deconv_refined[i])
    for i in range(len(net.deconv_raw)):
        transfer_deconv(supernet.deconv_raw[i], net.deconv_raw[i])
    for i in range(len(net.deconv_bnrelu)):
        transfer_bn(supernet.deconv_bnrelu[i][0], net.deconv_bnrelu[i][0])
    for i in range(len(net.final_refined)):
        transfer_sep(supernet.final_refined[i], net.final_refined[i])
    for i in range(len(net.final_raw)):
        transfer_sep(supernet.final_raw[i], net.final_raw[i])

def diff(a, b):
    print(((a-b)**2).mean())

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
    cfg.freeze()
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    arch_manager = ArchManager(cfg)
    transfer_model = eval('models.pose_mobilenet.get_pose_net')(
        cfg, is_train=True, cfg_arch = fixed_arch
        )
    model = eval('models.pose_supermobilenet.get_pose_net')(
        cfg, is_train=True
        )
    # set eval mode
    model.eval()
    transfer_model.eval()
    # set super config
    model.arch_manager.is_search = True
    model.arch_manager.search_arch = fixed_arch

    dump_input = torch.randn(
        (1, 3, cfg.DATASET.INPUT_SIZE, cfg.DATASET.INPUT_SIZE)
    ).cuda()
    # print(get_model_summary(cfg.DATASET.INPUT_SIZE, transfer_model, dump_input))
    # return 
    if cfg.FP16.ENABLED:
        model = network_to_half(model)
    print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
    need_state_dict = {}
    state_dict = torch.load(cfg.TEST.MODEL_FILE)
    for key, value in state_dict.items():
        # if 'deconv' in key:
        #     continue
        # if 'final' in key:
        #     continue
        if key[:2] == '1.':
            key = key[2:]
        need_state_dict[key] = value
    model.load_state_dict(need_state_dict, strict=True)
    # Note Here! Needs Transfer?
    # model.re_organize_weights()
    print("re-organize success!")
    transfer(model, transfer_model, fixed_arch)
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    transfer_model = torch.nn.DataParallel(transfer_model, device_ids=cfg.GPUS).cuda()
    with torch.no_grad():
        output_transfer = transfer_model(dump_input)
        output = model(dump_input)
        # debug
        # output = model.module[1].first[0](dump_input.half())
        # output = model.module[1].first[1](output)
        # output = model.module[1].first[2](output, fixed_arch['input_channel'])
        # output_transfer = transfer_model.module.first[0](dump_input)
        # output_transfer = transfer_model.module.first[1](output_transfer)
        # output_transfer = transfer_model.module.first[2](output_transfer)
    print(output_transfer[0].shape)
    print(output[0].shape)
    diff(output[0], output_transfer[0])
    torch.save(transfer_model.module.state_dict(), './pretrain/crowdpose-XS.pth.tar')
    # model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()


if __name__ == '__main__':
    main()
