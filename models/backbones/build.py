from .resnet import build_resnet_backbone
from .osnet import build_osnet_backbone
from .resnest import build_resnest_backbone
from .resnext import build_resnext_backbone
from .shufflenet import build_shufflenetv2_backbone
from .mobilenet import build_mobilenetv2_backbone


def build_backbone(cfg):
    backbone_name = cfg.backbone_name
    if cfg.backbone_name == 'resnet':
        backbone = build_resnet_backbone(cfg)
    if cfg.backbone_name == 'osnet':
        backbone = build_osnet_backbone(cfg)
    if cfg.backbone_name == 'resnest':
        backbone = build_resnest_backbone(cfg)
    if cfg.backbone_name == 'resnext':
        backbone = build_resnext_backbone(cfg)
    if cfg.backbone_name == 'shufflenetv2':
        backbone = build_shufflenetv2_backbone(cfg)
    if cfg.backbone_name == 'mobilenetv2':
        backbone = build_mobilenetv2_backbone(cfg)
    return backbone

