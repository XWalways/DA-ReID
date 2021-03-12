from .head import Head

def build_heads(cfg, **kwargs):
    """
    Build REIDHeads defined by `cfg.MODEL.REID_HEADS.NAME`.
    """
    return Head(cfg, **kwargs)
