from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import time
import fast_utils.plugins

class Params(object):
    def __init__(self, cfg):
        self.num_joints = cfg.DATASET.NUM_JOINTS
        self.max_num_people = cfg.DATASET.MAX_NUM_PEOPLE

        self.detection_threshold = cfg.TEST.DETECTION_THRESHOLD
        self.tag_threshold = cfg.TEST.TAG_THRESHOLD
        self.use_detection_val = cfg.TEST.USE_DETECTION_VAL
        self.ignore_too_much = cfg.TEST.IGNORE_TOO_MUCH
        self.window_size = cfg.TEST.NMS_KERNEL

        if cfg.DATASET.WITH_CENTER and cfg.TEST.IGNORE_CENTER:
            self.num_joints -= 1

        if cfg.DATASET.WITH_CENTER and not cfg.TEST.IGNORE_CENTER:
            self.joint_order = [
                i - 1 for i in [18, 1, 2, 3, 4, 5, 6, 7, 12, 13, 8, 9, 10, 11, 14, 15, 16, 17]
            ]
        else:
            self.joint_order = [
                i - 1 for i in [1, 2, 3, 4, 5, 6, 7, 12, 13, 8, 9, 10, 11, 14, 15, 16, 17]
            ]

class HeatmapParser(object):
    def __init__(self, cfg):
        self.params = Params(cfg)
        self.tag_per_joint = cfg.MODEL.TAG_PER_JOINT

    def parse(self, det, tmap, scale):
        # Remove adjust & refine for fast inference
        tmap = tmap[:,:,:,:,0]
        count, val, tag, ind = fast_utils.plugins.find_peaks(det, tmap, self.params.detection_threshold, self.params.window_size, self.params.max_num_people)
        count, val, tag, ind = count[0], val[0], tag[0], ind[0]
        # print(tmap.shape, val.shape, tag.shape, ind.shape)
        num, ans = fast_utils.plugins.assign(count, val, tag, ind, torch.tensor(self.params.joint_order).int(), self.params.tag_threshold, self.params.max_num_people)
        ans = ans[:num]
        ans[:, :, :2] *= scale
        # print(peak_counts)
        return ans