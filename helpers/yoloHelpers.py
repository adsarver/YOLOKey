import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Helper Functions & Basic Blocks ---

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape for convolution."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with SiLU activation."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Method for fusing conv and bn layers for faster inference."""
        return self.act(self.conv(x))

class RepConv(nn.Module):
    """
    RepConv is a RepVGG-style block.
    This implementation is designed to be fuseable for deployment.
    """
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False):
        super(RepConv, self).__init__()
        self.deploy = deploy
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        assert k == 3 and p is None
        p = k // 2
        
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=p-k//2, g=g, act=False)

        # This is the key change: Only create the identity branch if channels match
        self.identity = nn.BatchNorm2d(c1) if c1 == c2 else None

    def forward(self, x):
        if self.deploy:
            # This is for fused deployment, not used in training
            return self.act(self.fused_conv(x))
        else:
            # Add the main 3x3 and 1x1 branches
            main_out = self.conv1(x)
            res_out = self.conv2(x)
            
            # Add the identity branch ONLY if it exists
            if self.identity is not None:
                identity_out = self.identity(x)
                return self.act(main_out + res_out + identity_out)
            else:
                return self.act(main_out + res_out)

    def forward_fuse(self, x):
        """Inference-time forward pass."""
        return self.act(self.conv(x))
    
    def get_equivalent_kernel_bias(self):
        """Derives the fused kernel and bias from the individual layers."""
        kernel3x3, bias3x3 = self._get_kernel_bias(self.conv1)
        kernel1x1, bias1x1 = self._get_kernel_bias(self.conv2)
        kernelid, biasid = self._get_kernel_bias(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _get_kernel_bias(self, layer):
        if isinstance(layer, nn.Conv2d):
            kernel = layer.weight
            bias = layer.bias if layer.bias is not None else 0
        elif isinstance(layer, nn.BatchNorm2d):
            running_mean = layer.running_mean
            running_var = layer.running_var
            gamma = layer.weight
            beta = layer.bias
            eps = layer.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return torch.zeros(self.c1, self.c1, 3, 3).to(gamma.device), beta - running_mean * gamma / std
        else: # Identity
            return torch.zeros(self.c1, self.c1, 3, 3).to(self.conv1.weight.device), torch.zeros(self.c1).to(self.conv1.weight.device)

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def fuse_convs(self):
        """Fuses the layers into a single convolution for deployment."""
        if self.deploy:
            return
        fused_kernel, fused_bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(self.c1, self.c2, 3, 1, 1, groups=self.g, bias=True).requires_grad_(False)
        self.conv.weight.data = fused_kernel
        self.conv.bias.data = fused_bias
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        self.deploy = True


class SPPELAN(nn.Module):
    """Spatial Pyramid Pooling - Enhanced Layer Aggregation Network."""
    def __init__(self, c1, c2, c_spp=256):
        super().__init__()
        self.c_spp = c_spp
        self.cv1 = Conv(c1, c_spp, 1, 1)
        self.cv2 = Conv(c1, c_spp, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=5, stride=1, padding=2) for _ in range(3)])
        self.cv_spp = nn.ModuleList([Conv(c1, c_spp, 1, 1) for _ in range(3)])

        concat_channels = c_spp * 5
        self.cv3 = Conv(concat_channels, c2, 1, 1)

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        y_spp_raw = [m(x) for m in self.m]
        y_spp = [self.cv_spp[i](y_spp_raw[i]) for i in range(len(y_spp_raw))]
        return self.cv3(torch.cat((y1, y2, *y_spp), 1))


class RepNCSPELAN4(nn.Module):
    """RepNCSP-ELAN Block, a key component of YOLOv9's neck."""
    def __init__(self, c1, c2, c3, c4, c5=1):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c1, c3, 1, 1)
        self.cv3 = nn.Sequential(
            RepConv(self.c, c4, 3, 1),
            Conv(c4, c4, 1, 1)
        )
        self.cv4 = nn.Sequential(
            RepConv(c4, c4, 3, 1),
            Conv(c4, c4, 1, 1)
        )
        self.cv5 = Conv(c3 + 2 * c4, c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))

class ADown(nn.Module):
    """Downsampling block with average pooling."""
    def __init__(self, c1, c2):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.pool = nn.AvgPool2d(2, stride=2)
        self.conv = Conv(c1, c2, 1, 1)

    def forward(self, x):
        return self.conv(self.pool(x))


class MDown(nn.Module):
    """Downsampling block with max pooling."""
    def __init__(self, c1, c2):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv = Conv(c1, c2, 1, 1)

    def forward(self, x):
        return self.conv(self.pool(x))