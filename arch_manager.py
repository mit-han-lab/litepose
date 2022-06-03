import numpy as np
import random
import copy

def rand(c):
	return random.randint(0, c - 1)

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ArchManager:
	def __init__(self, cfg):
		self.cfg = cfg
		self.expansion = [6]
		self.kernel_size = [7]
		self.input_channel = 24
		self.width_mult = [1.0, 0.75, 0.5, 0.25]
		self.deconv_setting = cfg.MODEL.EXTRA.NUM_DECONV_FILTERS
		self.is_search = False
		self.search_arch = None
		self.arch_setting = [
            # c, n, s
            [32, 4, 2],
            [64, 6, 2],
            [96, 8, 2],
            [160, 8, 1]
        ]

	def rand_kernel_size(self):
		l = len(self.kernel_size)
		return self.kernel_size[rand(l)]

	def rand_expansion(self):
		l = len(self.expansion)
		return self.expansion[rand(l)]
	
	def rand_channel(self, c):
		l = len(self.width_mult)
		new_c = c * self.width_mult[rand(l)]
		return _make_divisible(new_c, 8)

	def random_sample(self):
		if self.is_search == True:
			return self.search_arch
		cfg_arch = {}
		cfg_arch['img_size'] = 256 + 64 * rand(5)
		cfg_arch['input_channel'] = self.rand_channel(self.input_channel)
		cfg_arch['deconv_setting'] = []
		for i in range(len(self.deconv_setting)):
			cfg_arch['deconv_setting'].append(self.rand_channel(self.deconv_setting[i]))
		cfg_arch['backbone_setting'] = []
		for i in range(len(self.arch_setting)):
			stage = {}
			c, n, s = self.arch_setting[i]
			stage['num_blocks'] = n
			stage['stride'] = s
			stage['channel'] = self.rand_channel(c)
			stage['block_setting'] = []
			for j in range(stage['num_blocks']):
				stage['block_setting'].append([6, 7])
			cfg_arch['backbone_setting'].append(stage)
		return cfg_arch

	def fixed_sample(self, reso=256, ratio=0.5):
		cfg_arch = {}
		cfg_arch['img_size'] = reso
		cfg_arch['input_channel'] = _make_divisible(self.input_channel * ratio, 8)
		cfg_arch['deconv_setting'] = []
		for i in range(len(self.deconv_setting)):
			cfg_arch['deconv_setting'].append(_make_divisible(self.deconv_setting[i] * ratio, 8))
		cfg_arch['backbone_setting'] = []
		for i in range(len(self.arch_setting)):
			stage = {}
			c, n, s = self.arch_setting[i]
			stage['num_blocks'] = n
			stage['stride'] = s
			stage['channel'] = _make_divisible(c * ratio, 8)
			stage['block_setting'] = []
			for j in range(stage['num_blocks']):
				stage['block_setting'].append([6, 7])
			cfg_arch['backbone_setting'].append(stage)
		return cfg_arch

