"""
Feature Pyramid Network (FPN) and FCOS detection head.

FPN:
    Takes multi-scale backbone features (C3, C4, C5) and builds a
    top-down feature pyramid (P3, P4, P5) with uniform channel depth.

FCOS Head:
    For each pixel on each FPN level, predicts:
    - Classification logits (num_classes channels)
    - Bounding box regression (l, t, r, b distances from pixel to box edges)
    - Centerness score (how close the pixel is to the GT box center)

Reference: Tian et al., "FCOS: Fully Convolutional One-Stage Object Detection"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PANet(nn.Module):
    """
    Path Aggregation Network (PANet).

    Builds a top-down feature pyramid from backbone outputs:
        C5 → P5 (stride 32)
        C4 + upsample(P5) → P4 (stride 16)
        C3 + upsample(P4) → P3 (stride 8)

    Each level uses:
    - A 1x1 lateral connection to reduce backbone channels to fpn_channels
    - A 3x3 smooth convolution after element-wise addition to reduce aliasing
    """

    def __init__(self, in_channels_list, out_channels=256):
        """
        Args:
            in_channels_list: [C3_channels, C4_channels, C5_channels]
                              e.g. [128, 256, 512] for ResNet-18
            out_channels: number of output channels for all FPN levels
        """
        super().__init__()

        # Lateral connections (1x1 conv to match channel dimensions)
        self.lateral_c3 = nn.Conv2d(in_channels_list[0], out_channels, 1)
        self.lateral_c4 = nn.Conv2d(in_channels_list[1], out_channels, 1)
        self.lateral_c5 = nn.Conv2d(in_channels_list[2], out_channels, 1)

        # Smooth convolutions (3x3 to reduce aliasing from upsampling + addition)
        self.smooth_p3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Bottom-up downsample convolutions (PANet) with BatchNorm to prevent explosion
        self.downsample_n3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample_n4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Bottom-up smooth convolutions
        self.smooth_n4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_n5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, c3, c4, c5):
        """
        Build feature pyramid.

        Args:
            c3: (B, C3, H/8,  W/8)  from backbone layer2
            c4: (B, C4, H/16, W/16) from backbone layer3
            c5: (B, C5, H/32, W/32) from backbone layer4

        Returns:
            [P3, P4, P5] — list of feature maps, all with out_channels
        """
        # Top-down pathway: start from the coarsest level
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(
            p5, size=c4.shape[2:], mode='nearest'
        )
        p3 = self.lateral_c3(c3) + F.interpolate(
            p4, size=c3.shape[2:], mode='nearest'
        )

        # Smooth to reduce aliasing artifacts
        p3 = self.smooth_p3(p3)
        p4 = self.smooth_p4(p4)
        p5 = self.smooth_p5(p5)

        # Bottom-up pathway (PANet)
        n3 = p3
        n4 = p4 + self.downsample_n3(n3)
        n4 = self.smooth_n4(n4)
        
        n5 = p5 + self.downsample_n4(n4)
        n5 = self.smooth_n5(n5)

        return [n3, n4, n5]


class FCOSHead(nn.Module):
    """
    FCOS detection head (shared weights across all FPN levels).

    For each FPN pixel, predicts:
    1. Classification: which of the num_classes objects (if any)
    2. Regression: (l, t, r, b) distances to bounding box edges
    3. Centerness: how centered the pixel is within the GT box

    Architecture:
        Classification subnet: 4× (Conv3x3 → GN → ReLU) → Conv3x3 → C channels
        Regression subnet:     4× (Conv3x3 → GN → ReLU) → Conv3x3 → 4 channels
        Centerness:            branches from regression features → Conv3x3 → 1 channel

    Each FPN level has its own learnable scale factor for regression outputs.
    """

    def __init__(self, in_channels=256, num_classes=5, num_convs=4):
        """
        Args:
            in_channels: input channels from FPN (default 256)
            num_classes: number of object classes (5 for this dataset)
            num_convs: number of stacked conv layers in each subnet
        """
        super().__init__()
        self.num_classes = num_classes

        # ---- Classification subnet ----
        cls_layers = []
        for _ in range(num_convs):
            cls_layers.extend([
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.GroupNorm(32, in_channels),  # GN instead of BN for small batches
                nn.ReLU(inplace=True),
            ])
        self.cls_subnet = nn.Sequential(*cls_layers)
        self.cls_score = nn.Conv2d(in_channels, num_classes, 3, padding=1)

        # ---- Regression subnet ----
        reg_layers = []
        for _ in range(num_convs):
            reg_layers.extend([
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.GroupNorm(32, in_channels),
                nn.ReLU(inplace=True),
            ])
        self.reg_subnet = nn.Sequential(*reg_layers)
        self.reg_pred = nn.Conv2d(in_channels, 4, 3, padding=1)

        # ---- Centerness ----
        # Branches from the regression features (shared conv layers)
        self.centerness = nn.Conv2d(in_channels, 1, 3, padding=1)

        # ---- Per-level learnable scale ----
        # Each FPN level gets its own scale factor for regression outputs,
        # allowing the network to predict different distance ranges per level
        self.scales = nn.ParameterList(
            [nn.Parameter(torch.ones(1)) for _ in range(3)]
        )

        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights following FCOS paper conventions.

        Key: cls_score bias is initialized to -log((1 - pi) / pi) where pi=0.01
        so that initial predictions have low confidence (reduces false positives
        early in training, critical for Focal Loss to work well).
        """
        # Subnet conv layers: normal init
        for modules in [self.cls_subnet, self.reg_subnet]:
            for layer in modules:
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        # Classification head: Focal Loss prior initialization
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, bias_value)

        # Regression and centerness heads
        nn.init.normal_(self.reg_pred.weight, std=0.01)
        nn.init.zeros_(self.reg_pred.bias)
        nn.init.normal_(self.centerness.weight, std=0.01)
        nn.init.zeros_(self.centerness.bias)

    def forward(self, features):
        """
        Run detection head on FPN features.

        Args:
            features: list of [P3, P4, P5] feature maps from FPN

        Returns:
            list of (cls_logits, reg_pred, ctr_logits) tuples per level:
                cls_logits: (B, num_classes, H, W) — raw logits (apply sigmoid later)
                reg_pred:   (B, 4, H, W) — positive (l, t, r, b) distances
                ctr_logits: (B, 1, H, W) — raw centerness logit
        """
        results = []
        strides = [8, 16, 32]
        for level_idx, feature in enumerate(features):
            # Classification branch
            cls_feat = self.cls_subnet(feature)
            cls_logits = self.cls_score(cls_feat)

            # Regression branch
            reg_feat = self.reg_subnet(feature)
            # Use exp() and multiply by stride as in the original FCOS paper
            # This prevents dead gradients from ReLU and helps predict reasonable initial sizes
            reg_pred = torch.exp(self.scales[level_idx] * self.reg_pred(reg_feat)) * strides[level_idx]

            # Centerness branch (uses regression features)
            ctr_logits = self.centerness(reg_feat)

            results.append((cls_logits, reg_pred, ctr_logits))

        return results
