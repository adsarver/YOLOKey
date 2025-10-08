import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import yaml
from glob import glob

class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, transforms=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.images = sorted(glob(os.path.join(self.img_dir, '*.jpg')))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.images[idx].replace(".jpg", ".txt").replace("images", "labels")

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load annotations
        boxes = []
        
        with open(label_path, "r") as f:
            for line in f.readlines():
                class_label, x, y, w, h = map(float, line.strip().split())
                boxes.append([class_label, x, y, w, h])

        if self.transforms:
            image = self.transforms(image)
        return image, torch.tensor(boxes)
