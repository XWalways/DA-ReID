import torchvision.transforms as T

from .transforms import *
from .autoaugment import AutoAugment

def build_transforms(opt, is_train=True):
    res = []

    if is_train:
        size_train = opt.train_size#for person reid: [384, 128], for vehicle resid: [256, 256]

        # augmix augmentation
        do_augmix = opt.do_augmix
        augmix_prob = 0.5

        # auto augmentation
        do_autoaug = opt.do_autoaug
        autoaug_prob = 0.1

        # horizontal filp
        do_flip = opt.do_flip
        flip_prob = 0.5

        # padding
        do_pad = opt.do_pad
        padding = 10
        padding_mode = 'constant'

        # color jitter
        do_cj = opt.do_cj
        cj_prob = 0.5
        cj_brightness = 0.15
        cj_contrast = 0.15
        cj_saturation = 0.1
        cj_hue = 0.1

        # random affine
        do_affine = opt.do_affine

        # random erasing
        do_rea = opt.do_rea
        rea_prob = 0.5
        rea_value = [0.485*255, 0.456*255, 0.406*255]

        # random patch
        do_rpt = opt.do_rpt
        rpt_prob = 0.5

        if do_autoaug:
            res.append(T.RandomApply([AutoAugment()], p=autoaug_prob))

        res.append(T.Resize(size_train, interpolation=3))
        if do_flip:
            res.append(T.RandomHorizontalFlip(p=flip_prob))
        if do_pad:
            res.extend([T.Pad(padding, padding_mode=padding_mode), T.RandomCrop(size_train)])
        if do_cj:
            res.append(T.RandomApply([T.ColorJitter(cj_brightness, cj_contrast, cj_saturation, cj_hue)], p=cj_prob))
        if do_affine:
            res.append(T.RandomAffine(degrees=0, translate=None, scale=[0.9, 1.1], shear=None, resample=False,
                                      fillcolor=128))
        if do_augmix:
            res.append(AugMix(prob=augmix_prob))
        res.append(ToTensor())
        if do_rea:
            res.append(T.RandomErasing(p=rea_prob, value=rea_value))
        if do_rpt:
            res.append(RandomPatch(prob_happen=rpt_prob))
    else:
        size_test = opt.test_size
        res.append(T.Resize(size_test, interpolation=3))
        res.append(ToTensor())
    return T.Compose(res)
