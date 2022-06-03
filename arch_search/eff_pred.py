import os
import numpy as np
import torch
import torch.nn as nn
from ptflops import get_model_complexity_info
from arch_search.arch_gen import ArchGenerator

class EfficiencyEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.arch_gen = ArchGenerator(cfg)

    def predict_eff(self, cfg_arch):
        model = self.arch_gen.arch2model(self.cfg, cfg_arch)
        reso = cfg_arch['img_size']
        macs, params = get_model_complexity_info(model, (3, reso, reso), print_per_layer_stat=False,
                                                 as_strings=True, verbose=False)
        macs = float(macs.split(' ')[0])
        return macs