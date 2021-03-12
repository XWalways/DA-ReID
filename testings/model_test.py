import torch

import sys
sys.path.append('.')
from configs import cfg
from models.backbones import build_resnet_backbone
cfg.with_ibn = True
cfg.with_se = True
cfg.depth = '101x'

net = build_resnet_backbone(cfg)
