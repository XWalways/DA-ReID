import math
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn import Conv2d, ReLU
from torch.nn.modules.utils import _pair
from torch.nn import Parameter
import copy
import os

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class Non_local(nn.Module):
    def __init__(self, in_channels, bn_norm, reduc_ratio=2):
        super(Non_local, self,).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio // reduc_ratio

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            get_norm(bn_norm, self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
                :param x: (b, t, h, w)
                :return x: (b, t, h, w)
        """
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel / reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel / reduction), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight_freeze=False, bias_freeze=False, weight_init=1.0,
                 bias_init=0.0, **kwargs):
        super().__init__(num_features, eps=eps, momentum=momentum)
        if weight_init is not None: nn.init.constant_(self.weight, weight_init)
        if bias_init is not None: nn.init.constant_(self.bias, bias_init)
        self.weight.requires_grad_(not weight_freeze)
        self.bias.requires_grad_(not bias_freeze)

class IBN(nn.Module):
    def __init__(self, planes, bn_norm, **kwargs):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = get_norm(bn_norm, half2, **kwargs)
    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out

class GhostBatchNorm(BatchNorm):
    def __init__(self, num_features, num_splits=1, **kwargs):
        super().__init__(num_features, **kwargs)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            self.running_mean = self.running_mean.repeat(self.num_splits)
            self.running_var = self.running_var.repeat(self.num_splits)
            outputs = F.batch_norm(
                    input.view(-1, C * self.num_splits, H, W), self.running_mean, self.running_var,
                    self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                    True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0)
            return outputs
        else:
            return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, False, self.momentum, self.eps)


def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        main_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    if new_v < 0.9 * v:
        new_v += divisor
    
    return new_v

def get_norm(norm, out_channels, **kwargs):
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
                "BN": BatchNorm,
                "GhostBN": GhostBatchNorm,
                "GN": lambda channels, **args: nn.GroupNorm(32, channels),
                }[norm]
    return norm(out_channels, **kwargs)

class SplAtConv2d(nn.Module):
    """Split-Attention Conv2d
    """

    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4,
                 rectify=False, rectify_avg=False, norm_layer=None,
                 dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        if self.rectify:
            from rfconv import RFConv2d
            self.conv = RFConv2d(in_channels, channels * radix, kernel_size, stride, padding, dilation,
                                 groups=groups * radix, bias=bias, average_mode=rectify_avg, **kwargs)
        else:
            self.conv = Conv2d(in_channels, channels * radix, kernel_size, stride, padding, dilation,
                               groups=groups * radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = get_norm(norm_layer, channels * radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = get_norm(norm_layer, inter_channels)
        self.fc2 = Conv2d(inter_channels, channels * radix, 1, groups=self.cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            if torch.__version__ < '1.5':
                splited = torch.split(x, int(rchannel // self.radix), dim=1)
            else:
                splited = torch.split(x, rchannel // self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            if torch.__version__ < '1.5':
                attens = torch.split(atten, int(rchannel // self.radix), dim=1)
            else:
                attens = torch.split(atten, rchannel // self.radix, dim=1)
            out = sum([att * split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()


class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x
class DropBlock2D(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError



class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1, 1, 1)

class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'

class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    """ Same, but norm is trainable
    """

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = nn.Parameter(torch.ones(1) * norm)


class AdaptiveAvgMaxPool2d(nn.Module):
    def __init__(self):
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.gap = FastGlobalAvgPool2d()
        self.gmp = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        avg_feat = self.gap(x)
        max_feat = self.gmp(x)
        feat = avg_feat + max_feat
        return feat


class FastGlobalAvgPool2d(nn.Module):
    def __init__(self, flatten=False):
        super(FastGlobalAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)

class ClipGlobalAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = FastGlobalAvgPool2d()

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.clamp(x, min=0., max=1.)
        return x


class ArcSoftmax(nn.Module):
    def __init__(self, cfg, in_feat, num_classes):
        super().__init__()
        self.in_feat = in_feat
        self._num_classes = num_classes
        self.s = cfg.scale
        self.m = cfg.margin

        self.easy_margin = False

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.threshold = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

        self.weight = Parameter(torch.Tensor(num_classes, in_feat))
        nn.init.xavier_uniform_(self.weight)
        self.register_buffer('t', torch.zeros(1))

    def forward(self, features, targets):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=cosine.device)
        one_hot.scatter_(1, targets.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class CircleSoftmax(nn.Module):
    def __init__(self, cfg, in_feat, num_classes):
        super().__init__()
        self.in_feat = in_feat
        self._num_classes = num_classes
        self.s = cfg.scale
        self.m = cfg.margin

        self.weight = Parameter(torch.Tensor(num_classes, in_feat))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, features, targets):
        sim_mat = F.linear(F.normalize(features), F.normalize(self.weight))
        alpha_p = torch.clamp_min(-sim_mat.detach() + 1 + self.m, min=0.)
        alpha_n = torch.clamp_min(sim_mat.detach() + self.m, min=0.)
        delta_p = 1 - self.m
        delta_n = self.m

        s_p = self.s * alpha_p * (sim_mat - delta_p)
        s_n = self.s * alpha_n * (sim_mat - delta_n)

        targets = F.one_hot(targets, num_classes=self._num_classes)

        pred_class_logits = targets * s_p + (1.0 - targets) * s_n

        return pred_class_logits

class CosSoftmax(nn.Module):
    r"""Implement of large margin cosine distance:
    Args:
        in_feat: size of each input sample
        num_classes: size of each output sample
    """

    def __init__(self, cfg, in_feat, num_classes):
        super().__init__()
        self.in_features = in_feat
        self._num_classes = num_classes
        self.s = cfg.scale
        self.m = cfg.margin
        self.weight = Parameter(torch.Tensor(num_classes, in_feat))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, targets):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        targets = F.one_hot(targets, num_classes=self._num_classes)
        output = (targets * phi) + ((1.0 - targets) * cosine)
        output *= self.s

        return output

class Checkpointer(object):
    """
    A checkpointer that can save/load model as well as extra checkpointable
    objects.
    """
    def __init__(self, model, save_dir, **checkpointables):
        self.model = model
        self.checkpointables = copy.copy(checkpointables)
        self.save_dir = save_dir

    def save(self, name, **kwargs):
        data = {}
        data['model'] = self.model.state_dict()
        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        basename = "{}.pth".format(name)
        save_file = os.path.join(self.save_dir, basename)
        torch.save(data, save_file)
        self.tag_last_checkpoint(basename)

    def load(self, path, checkpointables):
        if not path:
            return {}
        if not os.path.isfile(path):
            assert os.path.isfile(path), "Checkpoint not found!"

        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        checkpoint_state_dict = checkpoint.pop("model")
        for k in list(checkpoint_state_dict.keys()):
            v = checkpoint_state_dict[k]
            if not isinstance(v, np.ndarray) and not isinstance(v, torch.Tensor):
                raise ValueError("Unsupported type found in checkpoint! {}: {}".format(k, type(v)))
            if not isinstance(v, torch.Tensor):
                checkpoint_state_dict[k] = torch.from_numpy(v)

        self.model.load_state_dict(checkpoint_state_dict)

        for key in self.checkpointables if checkpointables is None else checkpointables:
            if key in checkpoint:
                obj = self.checkpointables[key]
                obj.load_state_dict(checkpoint.pop(key))

        return checkpoint

    def resume_or_load(self, path, resume):
        if resume and os.path.exists(os.path.join(self.save_dir, "last_checkpoint")):
            return self.load(path)
        else:
            return self.load(path, checkpointables=[])

    def tag_last_checkpoint(self, last_filename_basenme):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, 'w'):
            f.write(last_filename_basename)

class PeriodicCheckpointer:
    def __init__(self, checkpointer, period, max_epoch):
        self.checkpointer = checkpointer
        self.period = int(period)
        self.max_epoch = max_epoch
        self.best_metric = -1

    def step(self, epoch, **kwargs):
        epoch = int(epoch)
        additional_state = {"epoch": epoch}
        additional_state.update(kwargs)

        if (epoch + 1) % self.period == 0 and epoch < self.max_epoch - 1:
            if additional_state["metric"] > self.best_metric:
                self.checkpointer.save(
                    "model_best", **additional_state
                )
                self.best_metric = additional_state["metric"]
            # Put it behind best model save to make last checkpoint valid
            self.checkpointer.save(
                "model_{:04d}".format(epoch), **additional_state
            )
        if epoch >= self.max_epoch - 1:
            if additional_state["metric"] > self.best_metric:
                self.checkpointer.save(
                    "model_best", **additional_state
                )
            self.checkpointer.save("model_final", **additional_state)

    def save(self, name, **kwargs):
        self.checkpointer.save(name, **kwargs)
