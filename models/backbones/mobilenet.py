import math

import torch
import torch.nn as nn
from ..common import get_norm, make_divisible

def conv_3x3_bn(inp, oup, stride, bn_norm):
    return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            get_norm(bn_norm, oup),
            nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup, bn_norm):
    return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            get_norm(bn_norm, oup),
            nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, bn_norm, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                get_norm(bn_norm, hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                get_norm(bn_norm, oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                get_norm(bn_norm, hidden_dim),
                nn.ReLU6(inplace=True),

                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                get_norm(bn_norm, hidden_dim),
                nn.ReLU6(inplace=True),

                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, bn_norm, width_mult=1.):
        super(MobileNetV2, self).__init__()

        self.cfgs = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2, bn_norm)]

        block = InvertedResidual

        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, bn_norm, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel, bn_norm)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



def build_mobilenetv2_backbone(cfg):
    pretrained = cfg.pretrained
    pretrain_path = cfg.pretrained_path
    bn_norm = cfg.backbone_norm
    depth = cfg.depth

    width_mult = {
        "1.0x": 1.0,
        "0.75x": 0.75,
        "0.5x": 0.5,
        "0.35x": 0.35,
        "0.25x": 0.25,
        "0.1x": 0.1,
    }[depth]

    model = MobileNetV2(bn_norm, width_mult)

    if pretrained:
        try:
            state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
        except FileNotFoundError as e:
            raise e
        except KeyError as e:
            raise e

    return model
