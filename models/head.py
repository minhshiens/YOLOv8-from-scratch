import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PANet(nn.Module):

    def __init__(self, in_channels_list, out_channels=256):
        
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

    def __init__(self, in_channels=256, num_classes=5, num_convs=4):
        
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
