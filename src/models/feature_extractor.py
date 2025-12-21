from __future__ import annotations
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, device="cpu", layers=("layer2", "layer3")):
        super().__init__()
        self.device = torch.device(device)
        self.layers = set(layers)

        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        self.conv1 = m.conv1
        self.bn1 = m.bn1
        self.relu = m.relu
        self.maxpool = m.maxpool
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4

        for p in self.parameters():
            p.requires_grad = False

        self.to(self.device)
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        x = x.to(self.device)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        out = {}
        if "layer1" in self.layers: out["layer1"] = f1
        if "layer2" in self.layers: out["layer2"] = f2
        if "layer3" in self.layers: out["layer3"] = f3
        if "layer4" in self.layers: out["layer4"] = f4
        return out
