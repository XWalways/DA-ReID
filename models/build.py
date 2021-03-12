import torch

def build_model(cfg, **kwargs):
    model_name = cfg.model_name
    if model_name == 'baseline':
        model = Baseline(cfg, **kwargs)
    if model_name == 'mgn':
        model = MGN(cfg, **kwargs)
    if model_name == 'moco':
        model = MoCo(cfg, **kwargs)
    if model_name == 'distiller':
        model = Distiller(cfg, **kwargs)
    return model

