import torch
from torch import nn
from .backbones import build_backbone
from .heads import build_heads
from .losses import *

class Baseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self._cfg = cfg

        assert len(cfg.pixel_mean) == len(cfg.pixel_std)

        self.register_buffer("pixel_mean", torch.tensor(cfg.pixel_mean).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(cfg.pixel_std).view(1, -1, 1, 1))

        self.backbone = build_backbone(cfg)

        self.heads = build_heads(cfg)


    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"].to(self.device)

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            outputs = self.heads(features, targets)
            losses = self.losses(outputs, targets)
            return losses
        else:
            outputs = self.heads(features)
            return outputs

    def preprocess_image(self, batched_inputs):
        """
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

    def losses(self, outputs, gt_labels):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        # fmt: off
        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs       = outputs['cls_outputs']
        pred_features     = outputs['features']
        # fmt: on

        # Log prediction accuracy
        accuracy = log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}
        loss_names = self._cfg.loss_names

        if "CrossEntropyLoss" in loss_names:
            loss_dict["loss_cls"] = cross_entropy_loss(
                cls_outputs,
                gt_labels,
                self._cfg.ce_epsilon,
                self._cfg.ce_alpha,
            ) * self._cfg.ce_scale

        if "TripletLoss" in loss_names:
            loss_dict["loss_triplet"] = triplet_loss(
                pred_features,
                gt_labels,
                self._cfg.tri_margin,
                self._cfg.tri_norm_feat,
                self._cfg.tri_hard_mining,
            ) * self._cfg.tri_scale

        if "CircleLoss" in loss_names:
            loss_dict["loss_circle"] = pairwise_circleloss(
                pred_features,
                gt_labels,
                self._cfg.circle.margin,
                self._cfg.circle.gamma,
            ) * self._cfg.circle.scale

        if "Cosface" in loss_names:
            loss_dict["loss_cosface"] = pairwise_cosface(
                pred_features,
                gt_labels,
                self._cfg.cosface.margin,
                self._cfg.cosface.gamma,
            ) * self._cfg.cosface.scale

        return accuracy, loss_dict
