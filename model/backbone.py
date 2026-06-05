"""
ResNet-50 backbone for feature extraction.

Implements the ResNet-50 architecture with Bottleneck blocks
(1x1, 3x3, 1x1 convolutions with expansion=4). Outputs multi-scale
feature maps at strides 8, 16, 32 for use with FPN.

Optionally loads pretrained ImageNet weights from torchvision.
"""

import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    """
    Bottleneck residual block for ResNet-50/101/152.
    
    Structure:
        input ──┬── Conv1x1 → BN → ReLU → Conv3x3 → BN → ReLU → Conv1x1 → BN ──┐
                │                                                              │ (+)→ ReLU
                └──── [optional 1x1 downsample] ───────────────────────────────┘
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet-18/34.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet18Backbone(nn.Module):
    """
    ResNet-18 backbone that extracts multi-scale features.

    Architecture (Blocks [2, 2, 2, 2]):
        Stem:   Conv7x7(s=2) → BN → ReLU → MaxPool(s=2)    → stride 4
        Layer1: 2x BasicBlock(64→64, s=1)                  → stride 4
        Layer2: 2x BasicBlock(64→128, s=2)    [C3 output]  → stride 8
        Layer3: 2x BasicBlock(128→256, s=2)   [C4 output]  → stride 16
        Layer4: 2x BasicBlock(256→512, s=2)   [C5 output]  → stride 32

    Returns (C3, C4, C5) feature maps for use with FPN.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        self.in_channels = 64

        # ---- Stem ----
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ---- Residual layers ----
        self.layer1 = self._make_layer(BasicBlock, 64, num_blocks=2, stride=1)    # stride 4
        self.layer2 = self._make_layer(BasicBlock, 128, num_blocks=2, stride=2)   # stride 8
        self.layer3 = self._make_layer(BasicBlock, 256, num_blocks=2, stride=2)   # stride 16
        self.layer4 = self._make_layer(BasicBlock, 512, num_blocks=2, stride=2)   # stride 32

        # Output channel dimensions for FPN (C3, C4, C5)
        self.out_channels = [128, 256, 512]

        # Initialize weights from scratch
        self._init_weights()

        # Optionally load pretrained ImageNet weights
        if pretrained:
            self._load_pretrained()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _load_pretrained(self):
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            pretrained_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        except (ImportError, AttributeError):
            from torchvision.models import resnet18
            pretrained_model = resnet18(pretrained=True)

        pretrained_dict = pretrained_model.state_dict()
        own_dict = self.state_dict()

        matched = {
            k: v for k, v in pretrained_dict.items()
            if k in own_dict and v.shape == own_dict[k].shape
        }

        own_dict.update(matched)
        self.load_state_dict(own_dict)
        print(f"[Backbone] Loaded {len(matched)}/{len(own_dict)} pretrained ResNet-18 weights")

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return c3, c4, c5


class ResNet50Backbone(nn.Module):
    """
    ResNet-50 backbone that extracts multi-scale features.

    Architecture (Blocks [3, 4, 6, 3]):
        Stem:   Conv7x7(s=2) → BN → ReLU → MaxPool(s=2)    → stride 4
        Layer1: 3x Bottleneck(64→256, s=1)                 → stride 4
        Layer2: 4x Bottleneck(256→512, s=2)   [C3 output]  → stride 8
        Layer3: 6x Bottleneck(512→1024, s=2)  [C4 output]  → stride 16
        Layer4: 3x Bottleneck(1024→2048, s=2) [C5 output]  → stride 32

    Returns (C3, C4, C5) feature maps for use with FPN.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        self.in_channels = 64

        # ---- Stem ----
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ---- Residual layers ----
        self.layer1 = self._make_layer(Bottleneck, 64, num_blocks=3, stride=1)    # stride 4
        self.layer2 = self._make_layer(Bottleneck, 128, num_blocks=4, stride=2)   # stride 8
        self.layer3 = self._make_layer(Bottleneck, 256, num_blocks=6, stride=2)   # stride 16
        self.layer4 = self._make_layer(Bottleneck, 512, num_blocks=3, stride=2)   # stride 32

        # Output channel dimensions for FPN (C3, C4, C5)
        self.out_channels = [512, 1024, 2048]

        # Initialize weights from scratch
        self._init_weights()

        # Optionally load pretrained ImageNet weights
        if pretrained:
            self._load_pretrained()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _load_pretrained(self):
        try:
            from torchvision.models import resnet50, ResNet50_Weights
            pretrained_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        except (ImportError, AttributeError):
            from torchvision.models import resnet50
            pretrained_model = resnet50(pretrained=True)

        pretrained_dict = pretrained_model.state_dict()
        own_dict = self.state_dict()

        matched = {
            k: v for k, v in pretrained_dict.items()
            if k in own_dict and v.shape == own_dict[k].shape
        }

        own_dict.update(matched)
        self.load_state_dict(own_dict)
        print(f"[Backbone] Loaded {len(matched)}/{len(own_dict)} pretrained ResNet-50 weights")

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return c3, c4, c5
