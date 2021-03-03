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

    if opt.data_path != 'msmt17':
        dataset = DataSet(opt.data_path)
    else:
        dataset = MSMT17(opt.data_path)

    num_classses = dataset.num_train_pids
    train_set = ImageDataset(dataset.train, train_transform)
    train_set_woEr = ImageDataset(dataset.train, train_transform_woEr)
    test_set = ImageDataset(dataset.gallery, test_transform)
    query_set = ImageDataset(dataset.query, test_transform)
    
    if opt.triplet:
        train_loader = DataLoader(
                train_set,
                sampler=RandomIdentitySampler(train_set, batch_id=opt.batchid, batch_image=opt.batchimage),
                batch_size=opt.batchid * opt.batchimage, num_workers=opt.num_workers, pin_memory=True)
        
        train_loader_woEr = DataLoader(
                train_set_woEr,
                sampler=RandomIdentitySampler(train_set_woEr, batch_id=opt.batchid, batch_image=opt.batchimage),
                batch_size=opt.batchid * opt.batchimage, num_workers=opt.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(
                train_set,
                batch_size=opt.batchid*opt.batchimage, num_workers=opt.num_workers, pin_memory=True)
        train_loader_woEr = DataLoader(
                train_set_woEr,
                batch_size=opt.batchid*opt.batchimage, num_workers=opt.num_workers, pin_memory=True)

    test_loader = DataLoader(
        test_set, batch_size=opt.batchtest, num_workers=opt.num_workers, pin_memory=True)
    query_loader = DataLoader(
        query_set, batch_size=opt.batchtest, num_workers=opt.num_workers, pin_memory=True)

    return train_loader, train_loader_woEr, test_loader, query_loader, dataset


