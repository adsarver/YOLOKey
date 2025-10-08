import torch
import torch.nn as nn

def autopad(k, p=None):  # kernel, padding
    """Pad to 'same' shape for convolution."""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class ConvBlock(nn.Module):
    """A block of Conv2D -> BatchNorm -> ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class YOLOBackbone(nn.Module):
    def __init__(self):
        super(YOLOBackbone, self).__init__()
        self.layers = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        return self.layers(x)

class YOLOHead(nn.Module):
    def __init__(self, grid_size, num_classes, num_anchors):
        super(YOLOHead, self).__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.detector = nn.Conv2d(128, num_anchors * (5 + num_classes), kernel_size=1)

    def forward(self, x):
        return self.detector(x).permute(0, 2, 3, 1).contiguous()
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
class ADown(nn.Module):
    """Downsampling block with average pooling."""
    def __init__(self, c1, c2):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.pool = nn.AvgPool2d(2, stride=2)
        self.conv = ConvBlock(c1, c2, 1, 1)

    def forward(self, x):
        return self.conv(self.pool(x))


class MDown(nn.Module):
    """Downsampling block with max pooling."""
    def __init__(self, c1, c2):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv = ConvBlock(c1, c2, 1, 1)

    def forward(self, x):
        return self.conv(self.pool(x))