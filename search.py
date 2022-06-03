
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

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
from arch_search.evolution import EvolutionFinder
from arch_search.acc_pred import AccuracyEvaluator
from arch_search.eff_pred import EfficiencyEvaluator
from arch_manager import ArchManager
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    update_config(cfg, args)
    check_config(cfg)

    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    if cfg.MODEL.NAME == 'pose_mobilenet':
        arch_manager = ArchManager(cfg)
        cfg_arch = arch_manager.fixed_sample()
        model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
            cfg, is_train=False, cfg_arch = cfg_arch
        )
    else:
        model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
            cfg, is_train=False
        )

    if cfg.FP16.ENABLED:
        model = network_to_half(model)

    if cfg.TEST.MODEL_FILE:
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'model_best.pth.tar'
        )
        model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    acc_pred = AccuracyEvaluator(cfg, model)
    eff_pred = EfficiencyEvaluator(cfg)
    arch_manager = ArchManager(cfg)

    begin = time.time()
    cfg_arch = arch_manager.fixed_sample(reso=512, ratio=1.0)
    output_acc = acc_pred.predict_acc(cfg_arch)
    output_eff = eff_pred.predict_eff(cfg_arch)
    print("normal: acc:{}, eff:{}, time:{}".format(output_acc, output_eff, time.time() - begin))

    begin = time.time()
    cfg_arch = arch_manager.fixed_sample(reso=256, ratio=0.5)
    output_acc = acc_pred.predict_acc(cfg_arch)
    output_eff = eff_pred.predict_eff(cfg_arch)
    print("normal: acc:{}, eff:{}, time:{}".format(output_acc, output_eff, time.time() - begin))

    # Begin Search
    arch_selector = EvolutionFinder(cfg, eff_pred, acc_pred)
    arch_selector.set_efficiency_constraint(8)
    best_valids = arch_selector.run_evolution_search(verbose=True)
    print('Get point: (eff, acc) = ({}, {})'.format(best_valids[2], best_valids[0]))
    # add data
    if(os.path.exists('arch_search/result') == False):
        os.makedirs('arch_search/result')
    dict = {}
    dict[0] = (arch_selector.efficiency_constraint, best_valids)
    with open('arch_search/result/search_result.json', 'w') as f:
        json.dump(dict, f)



if __name__ == '__main__':
    main()