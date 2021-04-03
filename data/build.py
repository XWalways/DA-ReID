import torch
from .transforms import RandomErasing
from torch.utils.data import DataLoader
from torchvision import transforms as T
from .datasets import DataSet, ImageDataset, MSMT17
from .samplers import RandomIdentitySampler 
import os
import re

#for market-1501, dukemtmc, cuhk03, msmt17 datasets 
def make_data_loader(opt):
    train_transform = T.Compose([
        T.Resize((384, 128), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        RandomErasing(probability=0.5, mean=[0.0, 0.0, 0.0])
    ])

    train_transform_woEr = T.Compose([
        T.Resize((384, 128), interpolation=3),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = T.Compose([
        T.Resize((384, 128), interpolation=3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if 'MSMT' not in opt.data_path:
        dataset = DataSet(opt.data_path)
    else:
        dataset = MSMT17(opt.data_path)

    num_classses = dataset.num_train_pids
    train_set = ImageDataset(dataset.train, train_transform)
    train_set_woEr = ImageDataset(dataset.train, train_transform_woEr)
    test_set = ImageDataset(dataset.gallery, test_transform)
    query_set = ImageDataset(dataset.query, test_transform)
    
    train_loader = DataLoader(
            train_set,
            sampler=RandomIdentitySampler(train_set, batch_id=opt.batchid, batch_image=opt.batchimage),
            batch_size=opt.batchid * opt.batchimage, num_workers=opt.num_workers, pin_memory=True)
        
    train_loader_woEr = DataLoader(
            train_set_woEr,
            sampler=RandomIdentitySampler(train_set_woEr, batch_id=opt.batchid, batch_image=opt.batchimage),
            batch_size=opt.batchid * opt.batchimage, num_workers=opt.num_workers, pin_memory=True)

    test_loader = DataLoader(
        test_set, batch_size=opt.batchtest, num_workers=opt.num_workers, pin_memory=True)
    query_loader = DataLoader(
        query_set, batch_size=opt.batchtest, num_workers=opt.num_workers, pin_memory=True)

    return train_loader, train_loader_woEr, test_loader, query_loader, dataset

from .common import CommDataset
from .transforms import build_transforms
from .datasets import *
from torch._six import container_abcs, string_classes, int_classes
from .samplers import NaiveIdentitySampler

def build_reid_train_loader(cfg, mapper=None, **kwargs):
    train_items = list()

    for d in cfg.datasets:
        if d == 'msmt17':
            dataset = MSMT(root=cfg.dataset_root, combineall=cfg.combineall, **kwargs)
        if d == 'cuhk03':
            dataset = CUHK03(root=cfg.dataset_root, combineall=cfg.combineall, **kwargs)
        if d == 'dukemtmcreid':
            dataset = DukeMTMC(root=cfg.dataset_root, combineall=cfg.combineall, **kwargs)
        if d == 'market1501':
            dataset = Market1501(root=cfg.dataset_root, combineall=cfg.combineall, **kwargs)
        if d == 'veri':
            dataset = VeRi(root=cfg.dataset_root, combineall=cfg.combineall, **kwargs)
        if d == 'vehicleid':
            dataset = VehicleID(root=cfg.dataset_root, combineall=cfg.combineall, **kwargs)
        if d == 'smallvehicleid':
            dataset = SmallVehicleID(root=cfg.dataset_root, combineall=cfg.combineall, **kwargs)
        if d == 'mediumvehicleid':
            dataset = MediumVehicleID(root=cfg.dataset_root, combineall=cfg.combineall, **kwargs)
        if d == 'largevehicleid':
            dataset = LargeVehicleID(root=cfg.dataset_root, combineall=cfg.combineall, **kwargs)
        if d == 'veriwild':
            dataset = VeRiWild(root=cfg.dataset_root, combineall=cfg.combineall, **kwargs)
        if d == 'smallveriwild':
            dataset = SmallVeRiWild(root=cfg.dataset_root, combineall=cfg.combineall, **kwargs)
        if d == 'mediumveriwild':
            dataset = MediumVeRiWild(root=cfg.dataset_root, combineall=cfg.combineall, **kwargs)
        if d == 'largeveriwild':
            dataset = LargeVeriWild(root=cfg.dataset_root, combineall=cfg.combineall, **kwargs)
        dataset.show_train()
        train_items.extend(dataset.train)
    if mapper is not None:
        transforms = mapper
    else:
        transforms = build_transforms(cfg, is_train=True)

    train_set = CommDataset(train_items, transforms, relabel=True)

    num_workers = cfg.num_workers
    batch_id = cfg.batch_id
    batch_img = cfg.batch_image

    data_sampler = NaiveIdentitySampler(train_set.img_items, batch_id, batch_img)

    batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, batch_id*batch_img, True)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=fast_batch_collator,
        pin_memory=True,
    )

    return train_loader

def build_reid_test_loader(cfg, dataset_name, mapper=None, **kwargs):
    d = dataset_name
    if d == 'msmt17':
        dataset = MSMT(root=cfg.dataset_root, combineall=cfg.combineall, **kwargs)
    if d == 'cuhk03':
        dataset = CUHK03(root=cfg.dataset_root, combineall=cfg.combineall, **kwargs)
    if d == 'dukemtmcreid':
        dataset = DukeMTMC(root=cfg.dataset_root, combineall=cfg.combineall, **kwargs)
    if d == 'market1501':
        dataset = Market1501(root=cfg.dataset_root, combineall=cfg.combineall, **kwargs)
    if d == 'veri':
        dataset = VeRi(root=cfg.dataset_root, combineall=cfg.combineall, **kwargs)
    if d == 'vehicleid':
        dataset = VehicleID(root=cfg.dataset_root, combineall=cfg.combineall, **kwargs)
    if d == 'smallvehicleid':
        dataset = SmallVehicleID(root=cfg.dataset_root, combineall=cfg.combineall, **kwargs)
    if d == 'mediumvehicleid':
        dataset = MediumVehicleID(root=cfg.dataset_root, combineall=cfg.combineall, **kwargs)
    if d == 'largevehicleid':
        dataset = LargeVehicleID(root=cfg.dataset_root, combineall=cfg.combineall, **kwargs)
    if d == 'veriwild':
        dataset = VeRiWild(root=cfg.dataset_root, combineall=cfg.combineall, **kwargs)
    if d == 'smallveriwild':
        dataset = SmallVeRiWild(root=cfg.dataset_root, combineall=cfg.combineall, **kwargs)
    if d == 'mediumveriwild':
        dataset = MediumVeRiWild(root=cfg.dataset_root, combineall=cfg.combineall, **kwargs)
    if d == 'largeveriwild':
        dataset = LargeVeriWild(root=cfg.dataset_root, combineall=cfg.combineall, **kwargs)
    dataset.show_test()

    test_items = dataset.query + dataset.gallery

    if mapper is not None:
        transforms = mapper
    else:
        transforms = build_transforms(cfg, is_train=False)

    test_set = CommDataset(test_items, transforms, relabel=False)

    test_loader = DataLoader(test_set, batch_size=cfg.batch_test, num_workers=cfg.num_workers, collate_fn=fast_batch_collator, pin_memory=True)

    return test_loader, len(dataset.query)


def fast_batch_collator(batched_inputs):
    """
    A simple batch collator for most common reid tasks
    """
    elem = batched_inputs[0]
    if isinstance(elem, torch.Tensor):
        out = torch.zeros((len(batched_inputs), *elem.size()), dtype=elem.dtype)
        for i, tensor in enumerate(batched_inputs):
            out[i] += tensor
        return out

    elif isinstance(elem, container_abcs.Mapping):
        return {key: fast_batch_collator([d[key] for d in batched_inputs]) for key in elem}

    elif isinstance(elem, float):
        return torch.tensor(batched_inputs, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batched_inputs)
    elif isinstance(elem, string_classes):
        return batched_inputs
