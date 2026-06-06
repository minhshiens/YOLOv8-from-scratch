"""
FCOS Object Detector — assembles backbone, FPN, and detection head.

Architecture overview:
    Input Image (B, 3, 416, 416)
         │
    ┌────▼────┐
    │ ResNet-18│ ← Backbone (pretrained on ImageNet)
    │ Backbone │
    └─┬──┬──┬─┘
      │  │  │
     C3 C4 C5   ← Multi-scale features (strides 8, 16, 32)
      │  │  │
    ┌─▼──▼──▼─┐
    │   FPN    │ ← Feature Pyramid Network
    └─┬──┬──┬─┘
      │  │  │
     P3 P4 P5   ← Unified 256-channel features
      │  │  │
    ┌─▼──▼──▼─┐
    │FCOS Head │ ← Shared detection head
    └──────────┘
      │  │  │
    cls reg ctr  ← Per-pixel predictions at each level
"""

import torch
import torch.nn as nn

from model.backbone import ResNet18Backbone
from model.head import FPN, FCOSHead


class FCOSDetector(nn.Module):
    """
    FCOS (Fully Convolutional One-Stage) Object Detector.

    Combines:
    - ResNet-50 backbone for multi-scale feature extraction
    - FPN for building top-down feature pyramid
    - FCOS head for dense per-pixel predictions
    """

    def __init__(self, num_classes=5, pretrained_backbone=True, fpn_channels=256):
        """
        Args:
            num_classes: number of object classes (5 for this dataset)
            pretrained_backbone: if True, load ImageNet weights for ResNet-50
            fpn_channels: number of channels in FPN (default 256)
        """
        super().__init__()
        self.num_classes = num_classes
        self.strides = [8, 16, 32]  # FPN level strides

        # Backbone: ResNet-18
        self.backbone = ResNet18Backbone(pretrained=pretrained_backbone)

        # FPN: reduces all features to fpn_channels (256)
        self.fpn = FPN(
            in_channels_list=self.backbone.out_channels,  # [128, 256, 512]
            out_channels=fpn_channels,
        )

        # FCOS detection head (shared across all FPN levels)
        self.head = FCOSHead(
            in_channels=fpn_channels,
            num_classes=num_classes,
            num_convs=4,
        )

    def forward(self, images):
        """
        Forward pass.

        Args:
            images: (B, 3, H, W) normalized input image tensor

        Returns:
            list of (cls_logits, reg_pred, ctr_logits) tuples, one per FPN level:
                cls_logits: (B, num_classes, H_l, W_l) — raw logits
                reg_pred:   (B, 4, H_l, W_l) — positive ltrb distances
                ctr_logits: (B, 1, H_l, W_l) — raw centerness logits

            For inference, apply sigmoid to cls_logits and ctr_logits,
            then multiply: final_score = sigmoid(cls) * sigmoid(ctr)
        """
        # Extract multi-scale backbone features
        c3, c4, c5 = self.backbone(images)

        # Build top-down feature pyramid
        features = self.fpn(c3, c4, c5)  # [P3, P4, P5]

        # Run shared detection head on each FPN level
        predictions = self.head(features)

        return predictions
