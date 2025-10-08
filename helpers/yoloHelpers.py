import torch
import torch.nn as nn

def autopad(k, p=None):  # kernel, padding
    """Pad to 'same' shape for convolution."""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with SiLU activation."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
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
        p_3x3 = autopad(k, p) # Padding for 3x3 conv should be 1
        p_1x1 = autopad(1, None) # Padding for 1x1 conv should be 0

        self.conv1 = Conv(c1, c2, k, s, p=p_3x3, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=p_1x1, g=g, act=False)
        self.identity = nn.BatchNorm2d(c1) if c1 == c2 else None

    def forward(self, x):
        if hasattr(self, 'fused_conv'):
            return self.act(self.fused_conv(x))
        else:
            main_out = self.conv1(x)
            res_out = self.conv2(x)
            if self.identity is not None:
                identity_out = self.identity(x)
                return self.act(main_out + res_out + identity_out)
            else:
                return self.act(main_out + res_out)

    def fuse(self):
        """Inference-time forward pass."""
        return self.act(self.conv(x))

class ADown(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.cv1 = Conv(c1, c2 // 2, 3, 2)
        self.cv2 = nn.Sequential(
            nn.MaxPool2d(3, 2, 1),
            Conv(c1, c2 // 2, 1, 1)
        )

    def forward(self, x):
        return torch.cat([self.cv1(x), self.cv2(x)], 1)

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
    """RepNCSP-ELAN Block"""
    def __init__(self, c1, c2, c_hid, c_csp):
        super().__init__()
        # c1: input channels
        # c2: output channels
        # c_hid: hidden channels for the main path and input to the RepConv stack
        # c_csp: channels for the output of the RepConv stack
        
        self.cv1 = Conv(c1, c_hid, 1, 1) # Main branch
        self.cv2 = Conv(c1, c_hid, 1, 1) # Branch for RepConv stack
        
        self.m = nn.Sequential(
            RepConv(c_hid, c_csp, 3, 1),
            RepConv(c_csp, c_csp, 3, 1),
            RepConv(c_csp, c_csp, 3, 1),
            RepConv(c_csp, c_csp, 3, 1)
        )
        # The concatenation will have c_hid (from cv1) + c_csp (from m's output)
        self.cv3 = Conv(c_hid + c_csp, c2, 1, 1)

    def forward(self, x):
        # Path 1 (main branch)
        y1 = self.cv1(x)
        
        # Path 2 (stacked RepConv branch)
        y2 = self.cv2(x)
        y2 = self.m(y2)
        
        # Concatenate results from both paths and process
        return self.cv3(torch.cat((y2, y1), 1))

class Detect(nn.Module):
    """Detection head for YOLO models."""
    def __init__(self, nc=80, anchors=(), ch=()):
        super().__init__()
        self.stride = None  # Will be initialized by the model
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))
        
        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

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