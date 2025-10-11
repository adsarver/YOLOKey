import torch.nn as nn

## --- IGNORE ---
## This file is a placeholder to avoid import on weight transfer errors.

class Detect(nn.Module):
    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super().__init__()
        
class DDetect(nn.Module):
    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super().__init__()

class DualDetect(nn.Module):

    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super().__init__()
       
class DualDDetect(nn.Module):
    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super().__init__()

class TripleDetect(nn.Module):
    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super().__init__()
       
class TripleDDetect(nn.Module):
    def __init__(self, nc=80, ch=(), inplace=True):  # detection layer
        super().__init__()
        
class BaseModel(nn.Module):
    # YOLO base model
    def dummy(self, x):
        return x
    
class DetectionModel(BaseModel):
    # YOLO detection model
    def __init__(self, cfg='yolo.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()

Model = DetectionModel  # retain YOLO 'Model' class for backwards compatibility