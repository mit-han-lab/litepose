import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
from lib.models.layers.layers import InvBottleneck, convbnrelu, SepConv2d

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

class LitePose(nn.Module):
    def __init__(self, cfg, width_mult=1.0, round_nearest=8, cfg_arch=None):
        super(LitePose, self).__init__()
        # input_channel = 24
        # inverted_residual_setting = [
        #     # t, c, n, s
        #     [6, 32, 6, 2],
        #     [6, 64, 8, 2],
        #     [6, 96, 10, 2],
        #     [6, 160, 10, 1],
        # ]
        backbone_setting = cfg_arch['backbone_setting']
        input_channel = cfg_arch['input_channel']
        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.first = nn.Sequential(
            convbnrelu(3, 32, ker=3, stride=2),
            convbnrelu(32, 32, ker=3, stride=1, groups=32),
            nn.Conv2d(32, input_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(input_channel)
        )
        self.channel = [input_channel]
        # building inverted residual blocks
        self.stage = []
        for id_stage in range(len(backbone_setting)):
            n = backbone_setting[id_stage]['num_blocks']
            s = backbone_setting[id_stage]['stride']
            c = backbone_setting[id_stage]['channel']
            c = _make_divisible(c * width_mult, round_nearest)
            block_setting = backbone_setting[id_stage]['block_setting']
            layer = []
            for id_block in range(n):
                t, k = block_setting[id_block]
                stride = s if id_block == 0 else 1
                layer.append(InvBottleneck(input_channel, c, stride, ker=k, exp=t))
                input_channel = c
            layer = nn.Sequential(*layer)
            self.stage.append(layer)
            self.channel.append(c)
        self.stage = nn.ModuleList(self.stage)
        extra = cfg.MODEL.EXTRA
        self.filters = cfg_arch['deconv_setting']
        self.inplanes = self.channel[-1]
        self.deconv_refined, self.deconv_raw, self.deconv_bnrelu  = self._make_deconv_layers(
            extra.NUM_DECONV_LAYERS,
            self.filters,
            extra.NUM_DECONV_KERNELS,
        )
        self.final_refined, self.final_raw, self.final_channel = self._make_final_layers(cfg, self.filters)
        self.num_deconv_layers = extra.NUM_DECONV_LAYERS
        self.loss_config = cfg.LOSS

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_final_layers(self, cfg, num_filters):
        dim_tag = cfg.MODEL.NUM_JOINTS if cfg.MODEL.TAG_PER_JOINT else 1
        extra = cfg.MODEL.EXTRA
        final_raw= []
        final_refined = []
        final_channel = []
        for i in range(1, extra.NUM_DECONV_LAYERS):
            # input_channels = num_filters[i] + self.channel[-i-3]
            oup_joint = cfg.MODEL.NUM_JOINTS if cfg.LOSS.WITH_HEATMAPS_LOSS[i-1] else 0
            oup_tag = dim_tag if cfg.LOSS.WITH_AE_LOSS[i-1] else 0
            final_refined.append(SepConv2d(num_filters[i], oup_joint + oup_tag, ker=5))
            final_raw.append(SepConv2d(self.channel[-i-3], oup_joint + oup_tag, ker=5))
            final_channel.append(oup_joint + oup_tag)

        return nn.ModuleList(final_refined), nn.ModuleList(final_raw), final_channel

    def _make_deconv_layers(self, num_layers, num_filters, num_kernels):
        deconv_refined = []
        deconv_raw = []
        deconv_bnrelu = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])
            planes = num_filters[i]
            # inplanes = self.inplanes + self.channel[-i-2]
            layers = []
            deconv_refined.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            deconv_raw.append(
                nn.ConvTranspose2d(
                    in_channels=self.channel[-i-2],
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
            deconv_bnrelu.append(nn.Sequential(*layers))

        return nn.ModuleList(deconv_refined), nn.ModuleList(deconv_raw), nn.ModuleList(deconv_bnrelu)

    def forward(self, x):
        x = self.first(x)
        x_list = [x]
        for i in range(len(self.stage)):
            tmp = self.stage[i](x_list[-1])
            x_list.append(tmp)
        final_outputs = []
        input_refined = x_list[-1]
        input_raw = x_list[-2]
        for i in range(self.num_deconv_layers):
            next_input_refined = self.deconv_refined[i](input_refined)
            next_input_raw = self.deconv_raw[i](input_raw)
            input_refined= self.deconv_bnrelu[i](next_input_refined + next_input_raw)
            input_raw = x_list[-i-3]
            if i > 0:
                final_refined = self.final_refined[i-1](input_refined)
                final_raw = self.final_raw[i-1](input_raw)
                final_outputs.append(final_refined + final_raw)

        return final_outputs

def get_pose_net(cfg, is_train=False, cfg_arch=None):
    model = LitePose(cfg, cfg_arch=cfg_arch)
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        print(cfg.MODEL.PRETRAINED)
        if os.path.isfile(cfg.MODEL.PRETRAINED):
            print("load pre-train model")
            need_init_state_dict = {}
            state_dict = torch.load(cfg.MODEL.PRETRAINED, map_location=torch.device('cpu'))
            for key, value in state_dict.items():
                if 'deconv' in key:
                    continue
                if 'final' in key:
                    continue
                need_init_state_dict[key] = value
            try:
                model.load_state_dict(need_init_state_dict, strict=False)
            except:
                print("Error load!")
    return model