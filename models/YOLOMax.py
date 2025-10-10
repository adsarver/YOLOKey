import math
import torch
import torch.nn as nn
from utils.model_utils import Conv, RepNCSPELAN4, MConv, SPPELAN, ELAN1, Concat, DualDDetect, meta

# --- Main YOLOv9 Model ---

class YOLOMax(nn.Module):
    """
    YOLOv9 model implemented in PyTorch based on 
    https://github.com/WongKinYiu/yolov9/ YAML configuration.
    """
    def __init__(self, nc, ch=3):
        super().__init__()
        self.nc = nc
        self.hyp = meta  # Placeholder for hyperparameters if needed

        # --- Backbone ---
        self.b0 = Conv(ch, 16, 3, 2)                            # 0
        self.b1 = Conv(16, 32, 3, 2)                            # 1
        self.b2 = ELAN1(32, 32, 32, 16)                         # 2
        self.b3 = MConv(32, 64)                                 # 3
        self.b4 = RepNCSPELAN4(64, 64, 64, 32, 3)               # 4
        self.b5 = MConv(64, 96)                                 # 5
        self.b6 = RepNCSPELAN4(96, 96, 96, 48, 3)               # 6
        self.b7 = MConv(96, 128)                                # 7
        self.b8 = RepNCSPELAN4(128, 128, 128, 64, 3)            # 8

        # --- Head ---
        self.h9 = SPPELAN(128, 128, 64)                         # 9
        self.h10 = nn.Upsample(scale_factor=2, mode='nearest')  # 10
        self.h11 = Concat(1)                                    # 11
        self.h12 = RepNCSPELAN4(224, 96, 96, 48, 3)             # 12
        self.h13 = nn.Upsample(scale_factor=2, mode='nearest')  # 13
        self.h14 = Concat(1)                                    # 14
        self.h15 = RepNCSPELAN4(160, 64, 64, 32, 3)             # 15 (P3 for detection)
        self.h16 = MConv(64, 48)                                # 16
        self.h17 = Concat(1)                                    # 17
        self.h18 = RepNCSPELAN4(144, 96, 96, 48, 3)             # 18 (P4/16 for detection)
        self.h19 = MConv(96, 64)                                # 19
        self.h20 = Concat(1)                                    # 20
        self.h21 = RepNCSPELAN4(192, 128, 128, 64, 3)           # 21 (P5/32 for detection)
        self.h22 = SPPELAN(128, 128, 64)                        # 22
        self.h23 = nn.Upsample(scale_factor=2, mode='nearest')  # 23
        self.h24 = Concat(1)                                    # 24
        self.h25 = RepNCSPELAN4(224, 96, 96, 48, 3)             # 25
        self.h26 = nn.Upsample(scale_factor=2, mode='nearest')  # 26
        self.h27 = Concat(1)                                    # 27
        self.h28 = RepNCSPELAN4(160, 64, 64, 32, 3)             # 28

        ch_detect = [64, 96, 128, 64, 96, 128]
        self.h29 = DualDDetect(nc, ch=ch_detect)                # 29
        
        self.model = nn.ModuleList([
            self.b0, self.b1, self.b2, self.b3, self.b4, self.b5, self.b6, self.b7, self.b8, self.h9, # 0-9
            self.h10, self.h11, self.h12, self.h13, self.h14, self.h15, self.h16, self.h17, self.h18, self.h19, # 10-19
            self.h20, self.h21, self.h22, self.h23, self.h24, self.h25, self.h26, self.h27, self.h28, self.h29 #20-29
        ])

    def forward(self, x):
        x0 = self.b0(x)  # 0
        x1 = self.b1(x0) # 1
        x2 = self.b2(x1) # 2
        x3 = self.b3(x2) # 3
        x4 = self.b4(x3) # 4
        x5 = self.b5(x4) # 5
        x6 = self.b6(x5) # 6
        x7 = self.b7(x6) # 7
        x8 = self.b8(x7) # 8
        
        x9 = self.h9(x8) # 9
        x10 = self.h10(x9) # 10
        x11 = self.h11([x10, x6]) # 11
        x12 = self.h12(x11) # 12
        x13 = self.h13(x12) # 13
        x14 = self.h14([x13, x4]) # 14
        x15 = self.h15(x14) # 15 (P3 for detection)
        x16 = self.h16(x15) # 16
        x17 = self.h17([x16, x12]) # 17
        x18 = self.h18(x17) # 18 (P4/16 for detection)
        x19 = self.h19(x18) # 19
        x20 = self.h20([x19, x9]) # 20
        x21 = self.h21(x20) # 21 (P5/32 for detection)
        x22 = self.h22(x21) # 22
        x23 = self.h23(x22) # 23
        x24 = self.h24([x23, x6]) # 24
        x25 = self.h25(x24) # 25
        x26 = self.h26(x25) # 26
        x27 = self.h27([x26, x4]) # 27
        x28 = self.h28(x27) # 28
        
        return self.h29([x28, x25, x22, x15, x18, x21]) # 38 (P3, P4, P5 for detection)
        
