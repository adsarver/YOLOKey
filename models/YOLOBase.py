import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers.yoloHelpers import Conv, RepConv, SPPELAN, RepNCSPELAN4, Detect

class YOLOBase(nn.Module):
    """
    Based off YOLOv9n architecture with an integrated Detect head.
    Args:
        nc (int): number of classes.
        ch (int): number of input channels. Default is 3 (for RGB images).
    """
    def __init__(self, nc, ch=3):
        super().__init__()
        self.nc = nc # number of classes
        
        # Define anchors for the detection head
        anchors = [
            [10, 13, 16, 30, 33, 23],      # P3/8
            [30, 61, 62, 45, 59, 119],     # P4/16
            [116, 90, 156, 198, 373, 326]  # P5/32
        ]

        self.stem = Conv(3, 64, 3, 2)

        # --- Backbone ---
        self.backbone1 = nn.Sequential(
            Conv(64, 128, 3, 2),
            RepNCSPELAN4(128, 128, 64, 32)
        )
        self.backbone2 = nn.Sequential(
            Conv(128, 128, 3, 2),
            RepNCSPELAN4(128, 128, 64, 32)
        )
        self.backbone3 = nn.Sequential(
            Conv(128, 256, 3, 2),
            RepNCSPELAN4(256, 256, 128, 64)
        )
        self.backbone4 = nn.Sequential(
            Conv(256, 256, 3, 2),
            RepNCSPELAN4(256, 256, 128, 64)
        )
        self.spp = SPPELAN(256, 256, c_spp=128)

        # --- Neck (Feature Pyramid Network - FPN) ---
        self.neck1 = Conv(256, 128, 1, 1) # P5 to P5_latent
        self.neck2 = Conv(256, 128, 1, 1) # P4 to P4_latent
        
        self.neck3 = RepNCSPELAN4(256, 256, 128, 64) # P4_in = P4_latent (128) + upsample(P5_latent) (128)
        
        self.neck4 = Conv(256, 128, 1, 1) # P4_out to P4_up_latent
        self.neck5 = Conv(128, 128, 1, 1) # P3 to P3_latent

        self.neck6 = RepNCSPELAN4(256, 128, 64, 32) # P3_in = P3_latent (128) + upsample(P4_up_latent) (128)

        # --- Head (Path Aggregation Network - PAN) ---
        self.pan1 = Conv(128, 128, 3, 2) # P3_out to P3_down
        
        self.pan2 = RepNCSPELAN4(384, 256, 128, 64) # P4_pan_in = P3_down (128) + P4_out (256)

        self.pan3 = Conv(256, 256, 3, 2) # P4_pan_out to P4_down
        
        self.pan4 = RepNCSPELAN4(512, 256, 128, 64) # P5_pan_in = P4_down (256) + P5 (256)

        # --- Detection Head ---
        self.head = Detect(nc, anchors=anchors, ch=(128, 256, 256))

    def forward(self, x):
        # Backbone
        x2 = self.backbone1(self.stem(x))
        x3 = self.backbone2(x2)
        x4 = self.backbone3(x3)
        x5 = self.spp(self.backbone4(x4))

        # FPN Neck
        p5_latent = self.neck1(x5)
        p4_latent = self.neck2(x4)
        
        p5_upsampled = F.interpolate(p5_latent, size=p4_latent.shape[2:], mode='nearest')
        p4_out = self.neck3(torch.cat([p4_latent, p5_upsampled], 1))

        p4_up_latent = self.neck4(p4_out)
        p3_latent = self.neck5(x3)

        p4_upsampled = F.interpolate(p4_up_latent, size=p3_latent.shape[2:], mode='nearest')
        p3_out = self.neck6(torch.cat([p3_latent, p4_upsampled], 1))

        # PAN Head
        p3_downsampled = self.pan1(p3_out)
        p4_pan_out = self.pan2(torch.cat([p3_downsampled, p4_out], 1))

        p4_downsampled = self.pan3(p4_pan_out)
        p5_pan_out = self.pan4(torch.cat([p4_downsampled, x5], 1))
        
        # Pass feature maps to the detection head
        return self.head([p3_out, p4_pan_out, p5_pan_out])

    def fuse(self):
        """Fuse RepConv layers for faster inference."""
        for m in self.modules():
            if hasattr(m, 'fuse') and callable(m.fuse):
                m.fuse()
        return self

