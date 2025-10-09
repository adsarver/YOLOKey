import math
import torch
import torch.nn as nn
from helpers.yoloHelpers import Silence, Conv, RepNCSPELAN4, AConv, SPPELAN, Concat, CBLinear, CBFuse, DualDDetect, meta

# --- Main YOLOv9 Model ---

class YOLOBase(nn.Module):
    """
    YOLOv9 model implemented in PyTorch based on 
    https://github.com/WongKinYiu/yolov9/ YAML configuration.
    """
    def __init__(self, nc=80, ch=3):
        super().__init__()
        self.nc = nc
        self.hyp = meta  # Placeholder for hyperparameters if needed

        # --- Backbone ---
        self.b0 = Silence()                                     # 0
        self.b1 = Conv(ch, 32, 3, 2)                            # 1
        self.b2 = Conv(32, 64, 3, 2)                            # 2
        self.b3 = RepNCSPELAN4(64, 128, 128, 64, 1)                  # 3 
        self.b4 = AConv(128, 240)                                    # 4
        self.b5 = RepNCSPELAN4(240, 240, 240, 120, 1)                # 5
        self.b6 = AConv(240, 360)                                    # 6
        self.b7 = RepNCSPELAN4(360, 360, 360, 180, 1)                # 7
        self.b8 = AConv(360, 480)                                    # 8
        self.b9 = RepNCSPELAN4(480, 480, 480, 240, 1)                # 9

        # --- Head (FPN Path) ---
        self.h10 = SPPELAN(480, 480, 240)                            # 10
        self.h11 = nn.Upsample(scale_factor=2, mode='nearest')  # 11
        self.h12 = Concat(1)                                    # 12
        self.h13 = RepNCSPELAN4(840, 360, 360, 180, 1)         # 13, from cat(11, 7)
        self.h14 = nn.Upsample(scale_factor=2, mode='nearest')  # 14
        self.h15 = Concat(1)                                    # 15
        self.h16 = RepNCSPELAN4(600, 240, 240, 120, 1)         # 16, from cat(14, 5)

        # --- Head (PAN Path) ---
        self.h17 = AConv(240, 180)                                   # 17
        self.h18 = Concat(1)                                    # 18
        self.h19 = RepNCSPELAN4(540, 360, 360, 180, 1)         # 19, from cat(17, 13) 
        self.h20 = AConv(360, 240)                                   # 20
        self.h21 = Concat(1)                                    # 21
        self.h22 = RepNCSPELAN4(720, 480, 480, 240, 1)         # 22, from cat(20, 10)
        
        # --- Routing and Auxiliary Head ---
        self.h23 = CBLinear(240, [240])                         # 23, from 5
        self.h24 = CBLinear(360, [240, 360])                    # 24, from 7
        self.h25 = CBLinear(480, [240, 360, 480])               # 25, from 9

        self.h26 = Conv(ch, 32, 3, 2)                            # 26, from 0 (input)
        self.h27 = Conv(32, 64, 3, 2)                           # 27
        self.h28 = RepNCSPELAN4(64, 128, 128, 64, 1)                 # 28
        self.h29 = AConv(128, 240)                                   # 29
        self.h30 = CBFuse([0, 0, 0])                       # 30, from [23, 24, 25, -1]
    
        self.h31 = RepNCSPELAN4(240, 240, 240, 120, 1)               # 31 ------------------ STOPPED HERE
        self.h32 = AConv(240, 360)                                   # 32
        self.h33 = CBFuse([1, 1])                          # 33, from [24, 25, -1]
        
        self.h34 = RepNCSPELAN4(360, 360, 360, 180, 1)               # 34
        self.h35 = AConv(360, 480)                                   # 35
        self.h36 = CBFuse([2])                             # 36, from [25, -1]

        self.h37 = RepNCSPELAN4(480, 480, 480, 240, 1)               # 37

        # --- Detection Head ---
        # Input channels for DualDDetect from layers [31, 34, 37, 16, 19, 22]
        ch_detect = [240, 360, 480, 240, 360, 480]
        self.h38 = DualDDetect(nc, ch=ch_detect)                # 38
        
        
        self.model = nn.ModuleList([
            self.b0, self.b1, self.b2, self.b3, self.b4, self.b5, self.b6, self.b7, self.b8, self.b9, # 0-9
            self.h10, self.h11, self.h12, self.h13, self.h14, self.h15, self.h16, self.h17, self.h18, self.h19, # 10-19
            self.h20, self.h21, self.h22, self.h23, self.h24, self.h25, self.h26, self.h27, self.h28, self.h29, # 20-29
            self.h30, self.h31, self.h32, self.h33, self.h34, self.h35, self.h36, self.h37, self.h38  # 30-38
        ])

    def forward(self, x):
        # Store intermediate outputs for skip connections
        outputs = {}

        # --- Backbone ---
        outputs[0] = self.b0(x)
        x1 = self.b1(outputs[0])
        x2 = self.b2(x1)
        x3 = self.b3(x2)
        x4 = self.b4(x3)
        outputs[5] = self.b5(x4)
        x6 = self.b6(outputs[5])
        outputs[7] = self.b7(x6)
        x8 = self.b8(outputs[7])
        outputs[9] = self.b9(x8)

        # --- Head (FPN) ---
        outputs[10] = self.h10(outputs[9])
        x11 = self.h11(outputs[10])
        x12 = self.h12([x11, outputs[7]])
        outputs[13] = self.h13(x12)
        x14 = self.h14(outputs[13])
        x15 = self.h15([x14, outputs[5]])
        main_p3 = self.h16(x15) # Output for detection (layer 16)
        
        # --- Head (PAN) ---
        x17 = self.h17(main_p3)
        x18 = self.h18([x17, outputs[13]])
        main_p4 = self.h19(x18) # Output for detection (layer 19)
        x20 = self.h20(main_p4)
        x21 = self.h21([x20, outputs[10]])
        main_p5 = self.h22(x21) # Output for detection (layer 22)

        # --- Routing and Auxiliary Head ---
        cb_out_23 = self.h23(outputs[5])
        cb_out_24 = self.h24(outputs[7])
        cb_out_25 = self.h25(outputs[9])
        # cb_signals = cb_out_23 + cb_out_24 + cb_out_25

        x26 = self.h26(outputs[0])
        x27 = self.h27(x26)
        x28 = self.h28(x27)
        x29 = self.h29(x28)
        x30 = self.h30([cb_out_23, cb_out_24, cb_out_25, x29])
        
        aux_p3 = self.h31(x30) # Output for detection (layer 31)
        x32 = self.h32(aux_p3)
        x33 = self.h33([cb_out_24, cb_out_25, x32])

        aux_p4 = self.h34(x33) # Output for detection (layer 34)
        x35 = self.h35(aux_p4)
        x36 = self.h36([cb_out_25, x35])

        aux_p5 = self.h37(x36) # Output for detection (layer 37)
        
        # --- Detection ---
        preds = self.h38([aux_p3, aux_p4, aux_p5, main_p3, main_p4, main_p5])
        
        self.model = nn.ModuleList([
            self.b0, self.b1, self.b2, self.b3, self.b4, self.b5, self.b6, self.b7, self.b8, self.b9, # 0-9
            self.h10, self.h11, self.h12, self.h13, self.h14, self.h15, self.h16, self.h17, self.h18, self.h19, # 10-19
            self.h20, self.h21, self.h22, self.h23, self.h24, self.h25, self.h26, self.h27, self.h28, self.h29, # 20-29
            self.h30, self.h31, self.h32, self.h33, self.h34, self.h35, self.h36, self.h37, self.h38  # 30-38
        ])
        
        # The loss function expects a flat list of tensors. The traceback indicates it's receiving a 
        # nested list (likely [aux_preds, main_preds]). We flatten it here during training.
        if self.training:
            if isinstance(preds, (list, tuple)) and len(preds) > 0 and isinstance(preds[0], (list, tuple)):
                # Flatten the nested list into a single list of tensors
                return preds[0] + preds[1]
            return preds  # Return as-is if already flat
        else:  # Inference
            if isinstance(preds, (list, tuple)) and len(preds) > 0 and isinstance(preds[0], (list, tuple)):
                # For inference, return only the main predictions
                return preds[1]
            return preds
