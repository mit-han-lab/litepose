from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
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

class calibrate_tester:
    def __init__(self, cfg):
        self.cfg = cfg

    def test(self, model, cfg_arch):
        cfg = copy.deepcopy(self.cfg)
        cfg.defrost()
        reso = cfg_arch['img_size']
        cfg.DATASET.INPUT_SIZE = reso
        cfg.DATASET.OUTPUT_SIZE = [reso // 4, reso // 2]
        cfg.DATASET.TEST = 'search'
        cfg.DATASET.TRAIN = 'calibrate'
        cfg.freeze()
        model.module.arch_manager.is_search = True
        model.module.arch_manager.search_arch = cfg_arch
        data_loader, test_dataset = make_test_dataloader(cfg)
        train_data_loader, train_dataset = make_train_dataloader(cfg)
        model.train()
        for i, (images, heatmaps, masks, joints) in enumerate(train_data_loader):
            model(images)
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        parser = HeatmapParser(cfg)
        all_preds = []
        all_scores = []

        pbar = tqdm(total=len(test_dataset)) if cfg.TEST.LOG_PROGRESS else None
        #eval mode
        model.eval()
        for i, (images, annos) in enumerate(data_loader):
            assert 1 == images.size(0), 'Test batch size should be 1'
            image = images[0].cpu().numpy()
            # size at scale 1.0
            base_size, center, scale = get_multi_scale_size(
                image, cfg.DATASET.INPUT_SIZE, 1.0, min(cfg.TEST.SCALE_FACTOR)
            )
            with torch.no_grad():
                infer_begin = time.time()
                final_heatmaps = None
                tags_list = []
                for idx, s in enumerate(sorted(cfg.TEST.SCALE_FACTOR, reverse=True)):
                    input_size = cfg.DATASET.INPUT_SIZE
                    image_resized, center, scale = resize_align_multi_scale(
                        image, input_size, s, min(cfg.TEST.SCALE_FACTOR)
                    )
                    image_resized = transforms(image_resized)
                    image_resized = image_resized.unsqueeze(0).cuda()

                    outputs, heatmaps, tags = get_multi_stage_outputs(
                        cfg, model, image_resized, cfg.TEST.FLIP_TEST,
                        cfg.TEST.PROJECT2IMAGE,base_size
                    )

                    final_heatmaps, tags_list = aggregate_results(
                        cfg, s, final_heatmaps, tags_list, heatmaps, tags
                    )

                final_heatmaps = final_heatmaps / float(len(cfg.TEST.SCALE_FACTOR))
                tags = torch.cat(tags_list, dim=4)
                group_begin = time.time()
                grouped, scores = parser.parse(
                    final_heatmaps, tags, cfg.TEST.ADJUST, cfg.TEST.REFINE
                )
                final_results = get_final_preds(
                    grouped, center, scale,
                    [final_heatmaps.size(3), final_heatmaps.size(2)]
                )
                all_preds.append(final_results)
                all_scores.append(scores)

        
        os.system("mkdir ./tmp")
        name_values, _ = test_dataset.evaluate(
            cfg, all_preds, all_scores, "./tmp"
        )
        os.system("rm -rf ./tmp")
        return name_values['AP']

