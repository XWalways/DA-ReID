import sys
sys.path.append('.')
from data import build_reid_test_loader, build_reid_train_loader
from configs import cfg
cfg.datasets = ["market1501", "dukemtmcreid"]

build_reid_train_loader(cfg)
for dataset_name in cfg.datasets:
    build_reid_test_loader(cfg, dataset_name)
