import torch
import torch.nn as nn
from helpers.yoloHelpers import YOLOBackbone, YOLOHead

class YOLOBase(nn.Module):
    def __init__(self, grid_size=7, num_classes=20, num_anchors=3):
        super(YOLOBase, self).__init__()
        self.backbone = YOLOBackbone()
        self.head = YOLOHead(grid_size, num_classes, num_anchors)

    def forward(self, x):
        features = self.backbone(x)
        predictions = self.head(features)
        return predictions
