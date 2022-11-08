import os
import pickle
from typing import Tuple

import cv2
import numpy as np
import torch
import torchvision



from fast_utils.group import HeatmapParser
from core.inference import aggregate_results
from core.inference import get_multi_stage_outputs
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size
from utils.transforms import resize_align_multi_scale
from utils.vis import get_annotated_image
from torch.utils.dlpack import from_dlpack as torch_from_dlpack
from torch.utils.dlpack import to_dlpack as torch_to_dlpack

transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
)


def process(cfg, frame: np.ndarray, executor) -> np.ndarray:
    parser = HeatmapParser(cfg)
    print(type(frame))
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    img_res = min(h, w)
    image = image[h // 2 - img_res // 2:h // 2 + img_res // 2, w // 2 - img_res // 2:w // 2 + img_res // 2]
    # size at scale 1.0
    base_size, center, scale = get_multi_scale_size(
        image, cfg.DATASET.INPUT_SIZE, 1.0, min(cfg.TEST.SCALE_FACTOR)
    )
    res = 224
    base_size = (res, res)
    with torch.no_grad():
        final_heatmaps = None
        tags_list = []
        for idx, s in enumerate(sorted(cfg.TEST.SCALE_FACTOR, reverse=True)):
            input_size = cfg.DATASET.INPUT_SIZE
            image_resized, center, scale = resize_align_multi_scale(
                image, input_size, s, min(cfg.TEST.SCALE_FACTOR)
            )
            image_resized = transforms(image_resized)
            outputs, heatmaps, tags = get_multi_stage_outputs(
                cfg, executor, image_resized, cfg.TEST.FLIP_TEST,
                cfg.TEST.PROJECT2IMAGE, base_size
            )
            final_heatmaps, tags_list = aggregate_results(
                cfg, s, final_heatmaps, tags_list, heatmaps, tags
            )

        final_heatmaps = final_heatmaps / float(len(cfg.TEST.SCALE_FACTOR))
        tags = torch.cat(tags_list, dim=4)
        # grouped, scores = parser.parse(
        #     final_heatmaps, tags, cfg.TEST.ADJUST, cfg.TEST.REFINE
        # )
        # final_results = get_final_preds(
        #     grouped, center, scale,
        #     [final_heatmaps.size(3), final_heatmaps.size(2)]
        # )
        hmap = final_heatmaps.detach().cpu()
        tmap = tags.detach().cpu()
        final_results = parser.parse(
            hmap, tmap, img_res / res
        )
        output = get_annotated_image(image, final_results, dataset='CROWDPOSE')

    return output



def get_cfg(cfg_path):
    with open(cfg_path, 'rb') as f:
        cfg = pickle.load(f)
    cfg.defrost()
    cfg.DATASET.INPUT_SIZE = 448
    cfg.DATASET.OUTPUT_SIZE = [112, 224]
    cfg.TEST.FLIP_TEST = False
    cfg.TEST.ADJUST = False
    cfg.TEST.REFINE = False
    cfg.freeze()
    return cfg
