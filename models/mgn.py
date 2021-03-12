import copy
import torch
from torch import nn

from .common import get_norm
from .backbones import build_backbone
from .heads import build_heads
from .losses import *
from .backbones.resnet import Bottleneck

class MGN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        assert len(cfg.pixel_mean) == len(cfg.pixel_std)
        self.register_buffer("pixel_mean", torch.Tensor(cfg.pixel_mean).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.pixel_std).view(1, -1, 1, 1))

        # fmt: off
        # backbone
        bn_norm    = cfg.norm
        with_se    = cfg.with_se
        # fmt :on

        backbone = build_backbone(cfg)
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3[0]
        )
        res_conv4 = nn.Sequential(*backbone.layer3[1:])
        res_g_conv5 = backbone.layer4

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, bn_norm, False, with_se, downsample=nn.Sequential(
                nn.Conv2d(1024, 2048, 1, bias=False), get_norm(bn_norm, 2048))),
            Bottleneck(2048, 512, bn_norm, False, with_se),
            Bottleneck(2048, 512, bn_norm, False, with_se))
        res_p_conv5.load_state_dict(backbone.layer4.state_dict())

        # branch1
        self.b1 = nn.Sequential(
            copy.deepcopy(res_conv4),
            copy.deepcopy(res_g_conv5)
        )
        self.b1_head = build_heads(cfg)

        # branch2
        self.b2 = nn.Sequential(
            copy.deepcopy(res_conv4),
            copy.deepcopy(res_p_conv5)
        )
        self.b2_head = build_heads(cfg)
        self.b21_head = build_heads(cfg)
        self.b22_head = build_heads(cfg)

        # branch3
        self.b3 = nn.Sequential(
            copy.deepcopy(res_conv4),
            copy.deepcopy(res_p_conv5)
        )
        self.b3_head = build_heads(cfg)
        self.b31_head = build_heads(cfg)
        self.b32_head = build_heads(cfg)
        self.b33_head = build_heads(cfg)

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)  # (bs, 2048, 16, 8)

        # branch1
        b1_feat = self.b1(features)

        # branch2
        b2_feat = self.b2(features)
        b21_feat, b22_feat = torch.chunk(b2_feat, 2, dim=2)

        # branch3
        b3_feat = self.b3(features)
        b31_feat, b32_feat, b33_feat = torch.chunk(b3_feat, 3, dim=2)

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"].long().to(self.device)

            if targets.sum() < 0: targets.zero_()

            b1_outputs = self.b1_head(b1_feat, targets)
            b2_outputs = self.b2_head(b2_feat, targets)
            b21_outputs = self.b21_head(b21_feat, targets)
            b22_outputs = self.b22_head(b22_feat, targets)
            b3_outputs = self.b3_head(b3_feat, targets)
            b31_outputs = self.b31_head(b31_feat, targets)
            b32_outputs = self.b32_head(b32_feat, targets)
            b33_outputs = self.b33_head(b33_feat, targets)

            losses = self.losses(b1_outputs,
                        b2_outputs, b21_outputs, b22_outputs,
                        b3_outputs, b31_outputs, b32_outputs, b33_outputs,
                        targets)
            return losses
        else:
            b1_pool_feat = self.b1_head(b1_feat)
            b2_pool_feat = self.b2_head(b2_feat)
            b21_pool_feat = self.b21_head(b21_feat)
            b22_pool_feat = self.b22_head(b22_feat)
            b3_pool_feat = self.b3_head(b3_feat)
            b31_pool_feat = self.b31_head(b31_feat)
            b32_pool_feat = self.b32_head(b32_feat)
            b33_pool_feat = self.b33_head(b33_feat)

            pred_feat = torch.cat([b1_pool_feat, b2_pool_feat, b3_pool_feat, b21_pool_feat,
                                   b22_pool_feat, b31_pool_feat, b32_pool_feat, b33_pool_feat], dim=1)
            return pred_feat

    def preprocess_image(self, batched_inputs):
        r"""
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs["images"].cuda()
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs.cuda()
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self,
               b1_outputs,
               b2_outputs, b21_outputs, b22_outputs,
               b3_outputs, b31_outputs, b32_outputs, b33_outputs, gt_labels):
        # model predictions
        # fmt: off
        pred_class_logits = b1_outputs['pred_class_logits'].detach()
        b1_logits         = b1_outputs['cls_outputs']
        b2_logits         = b2_outputs['cls_outputs']
        b21_logits        = b21_outputs['cls_outputs']
        b22_logits        = b22_outputs['cls_outputs']
        b3_logits         = b3_outputs['cls_outputs']
        b31_logits        = b31_outputs['cls_outputs']
        b32_logits        = b32_outputs['cls_outputs']
        b33_logits        = b33_outputs['cls_outputs']
        b1_pool_feat      = b1_outputs['features']
        b2_pool_feat      = b2_outputs['features']
        b3_pool_feat      = b3_outputs['features']
        b21_pool_feat     = b21_outputs['features']
        b22_pool_feat     = b22_outputs['features']
        b31_pool_feat     = b31_outputs['features']
        b32_pool_feat     = b32_outputs['features']
        b33_pool_feat     = b33_outputs['features']
        # fmt: on

        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        b22_pool_feat = torch.cat((b21_pool_feat, b22_pool_feat), dim=1)
        b33_pool_feat = torch.cat((b31_pool_feat, b32_pool_feat, b33_pool_feat), dim=1)

        loss_dict = {}
        loss_names = self._cfg.loss_names

        if "CrossEntropyLoss" in loss_names:
            loss_dict['loss_cls_b1'] = cross_entropy_loss(
                b1_logits,
                gt_labels,
                self._cfg.ce_epsilon,
                self._cfg.ce_alpha,
            ) * self._cfg.ce_scale * 0.125
            loss_dict['loss_cls_b2'] = cross_entropy_loss(
                b2_logits,
                gt_labels,
                self._cfg.ce_epsilon,
                self._cfg.ce_alpha,
            ) * self._cfg.ce_scale * 0.125
            loss_dict['loss_cls_b21'] = cross_entropy_loss(
                b21_logits,
                gt_labels,
                self._cfg.ce_epsilon,
                self._cfg.ce_alpha,
            ) * self._cfg.ce_scale * 0.125
            loss_dict['loss_cls_b22'] = cross_entropy_loss(
                b22_logits,
                gt_labels,
                self._cfg.ce_epsilon,
                self._cfg.ce_alpha,
            ) * self._cfg.ce_scale * 0.125
            loss_dict['loss_cls_b3'] = cross_entropy_loss(
                b3_logits,
                gt_labels,
                self._cfg.ce_epsilon,
                self._cfg.ce_alpha,
            ) * self._cfg.ce_scale * 0.125
            loss_dict['loss_cls_b31'] = cross_entropy_loss(
                b31_logits,
                gt_labels,
                self._cfg.ce_epsilon,
                self._cfg.ce_alpha,
            ) * self._cfg.ce_scale * 0.125
            loss_dict['loss_cls_b32'] = cross_entropy_loss(
                b32_logits,
                gt_labels,
                self._cfg.ce_epsilon,
                self._cfg.ce_alpha,
            ) * self._cfg.ce_scale * 0.125
            loss_dict['loss_cls_b33'] = cross_entropy_loss(
                b33_logits,
                gt_labels,
                self._cfg.ce_epsilon,
                self._cfg.ce_alpha,
            ) * self._cfg.ce_scale * 0.125

        if "TripletLoss" in loss_names:
            loss_dict['loss_triplet_b1'] = triplet_loss(
                b1_pool_feat,
                gt_labels,
                self._cfg.tri_margin,
                self._cfg.tri_norm_feat,
                self._cfg.tri_hard_mining,
            ) * self._cfg.tri_scale * 0.2
            loss_dict['loss_triplet_b2'] = triplet_loss(
                b2_pool_feat,
                gt_labels,
                self._cfg.tri_margin,
                self._cfg.tri_norm_feat,
                self._cfg.tri_hard_mining,
            ) * self._cfg.tri_scale * 0.2
            loss_dict['loss_triplet_b3'] = triplet_loss(
                b3_pool_feat,
                gt_labels,
                self._cfg.tri_margin,
                self._cfg.tri_norm_feat,
                self._cfg.tri_hard_mining,
            ) * self._cfg.tri_scale * 0.2
            loss_dict['loss_triplet_b22'] = triplet_loss(
                b22_pool_feat,
                gt_labels,
                self._cfg.tri_margin,
                self._cfg.tri_norm_feat,
                self._cfg.tri_hard_mining,
            ) * self._cfg.tri_scale * 0.2
            loss_dict['loss_triplet_b33'] = triplet_loss(
                b33_pool_feat,
                gt_labels,
                self._cfg.tri_margin,
                self._cfg.norm_feat,
                self._cfg.tri_hard_mining,
            ) * self._cfg.tri_scale * 0.2

        return loss_dict
