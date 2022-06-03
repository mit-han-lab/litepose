import os
import numpy as np
import torch
import torch.nn as nn
from calibrate_test import calibrate_tester
import copy

class AccuracyEvaluator(nn.Module):

	def __init__(self, cfg, model):
		super(AccuracyEvaluator, self).__init__()
		self.cfg = cfg
		self.model = model
		self.calibrate_tester = calibrate_tester(cfg)

	def predict_acc(self, cfg_arch):
		model = copy.deepcopy(self.model)
		return self.calibrate_tester.test(model, cfg_arch)