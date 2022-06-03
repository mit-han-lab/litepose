import argparse
import os
import pprint
import shutil
import _init_paths
from config import cfg
from config import update_config

import dataset
import models

class ArchGenerator:
    def __init__(self, cfg):
        self.cfg = cfg
    def arch2model(self, cfg, cfg_arch):
        model = eval('models.pose_mobilenet.get_pose_net')(
            cfg, is_train=False, cfg_arch = cfg_arch
        )
        return model