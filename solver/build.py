import math

from . import lr_scheduler
from . import optim

def build_optimizer(cfg, model):

    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad: continue
        lr = cfg.base_lr

        weight_decay = cfg.weight_decay

        if "heads" in key:
            lr *= cfg.head_lr_factor
        if "bias" in key:
            lr *= cfg.bias_lr_factor
            weight_decay = cfg.weight_decay_bias

        params += [{"name": key, "params": [value], "lr": lr, "weight_decay": weight_decay}]

    solver_opt = cfg.solver_opt

    if solver_opt == "SGD":
        opt_fns = getattr(optim, solver_opt)(
            params,
            momentum=cfg.momentum,
            nesterov=True if cfg.momentum and cfg.nesterov else False
        )

    else:
        opt_fns = getattr(optim, solver_opt)(params)

    return opt_fns

def build_lr_scheduler(cfg, optimizer, iters_per_epoch):
    max_epoch = cfg.max_epoch - max(math.ceil(cfg.warmup_iters / iters_per_epoch), cfg.delay_epochs)

    scheduler_dict = {}

    scheduler_args = {
        "MultiStepLR": {
            "optimizer": optimizer,
            # multi-step lr scheduler options
            "milestones": cfg.lr_steps,
            "gamma": cfg.lr_gamma,
        },
        "CosineAnnealingLR": {
            "optimizer": optimizer,
            # cosine annealing lr scheduler options
            "T_max": max_epoch,
            "eta_min": cfg.eta_min_lr,
        },

    }

    scheduler_dict["lr_sched"] = getattr(lr_scheduler, cfg.lr_sched)(**scheduler_args[cfg.lr_sched])

    if cfg.warmup_iters > 0:
        warmup_args = {
            "optimizer": optimizer,
            "warmup_factor": cfg.warmup_factor,
            "warmup_iters": cfg.warmup_iters,
            "warmup_method": cfg.warmup_method,
        }
        scheduler_dict["warmup_sched"] = lr_scheduler.WarmupLR(**warmup_args)

    return scheduler_dict




