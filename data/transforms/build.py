import torchvision.transforms as T

from .transforms import *
from .autoaugment import AutoAugment

def build_transforms(cfg, is_train=True):
    res = []

    if is_train:
        size_train = [384, 128]

        # augmix augmentation
        do_augmix = False
        augmix_prob = 0.5

        # auto augmentation
        do_autoaug = True
        autoaug_prob = 0.1

        # horizontal filp
        do_flip = False
        flip_prob = 0.5

        # padding
        do_pad = False
        padding = 10
        padding_mode = 'constant'

        # color jitter
        do_cj = False
        cj_prob = 0.5
        cj_brightness = 0.15
        cj_contrast = 0.15
        cj_saturation = 0.1
        cj_hue = 0.1

        # random affine
        do_affine = False

        # random erasing
        do_rea = False
        rea_prob = 0.5
        rea_value = [0.485*255, 0.456*255, 0.406*255]

        # random patch
        do_rpt = False
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
        size_test = [384, 128]
        res.append(T.Resize(size_test, interpolation=3))
        res.append(ToTensor())
    return T.Compose(res)
