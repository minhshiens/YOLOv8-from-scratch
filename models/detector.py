import torch
import torch.nn as nn

from models.backbone import ResNet50Backbone
from models.head import PANet, FCOSHead

class FCOSDetector(nn.Module):

    def __init__(self, num_classes=5, pretrained_backbone=True, fpn_channels=256):
        
        super().__init__()
        self.num_classes = num_classes
        self.strides = [8, 16, 32]  # FPN level strides

        # Backbone: ResNet-50 → outputs C3 (512ch), C4 (1024ch), C5 (2048ch)
        self.backbone = ResNet50Backbone(pretrained=pretrained_backbone)

        # PANet: reduces all features to fpn_channels (256) and aggregates
        self.panet = PANet(
            in_channels_list=self.backbone.out_channels,
            out_channels=fpn_channels
        )

        # FCOS detection head (shared across all FPN levels)
        self.head = FCOSHead(
            in_channels=fpn_channels,
            num_classes=num_classes,
            num_convs=4,
        )

    def forward(self, images):
        
        # Extract multi-scale backbone features
        c3, c4, c5 = self.backbone(images)

        # Build top-down feature pyramid
        features = self.panet(c3, c4, c5)  # [P3, P4, P5]

        # Run shared detection head on each FPN level
        predictions = self.head(features)

        return predictions
