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
    
def yolo_collate_fn(batch):
    """
    Collate function to handle images and labels of different sizes.
    Pads images to the maximum size in the batch and pads labels with zeros.
    """
    # Separate images and labels from the batch
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Find the maximum image dimensions in the batch
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    
    # Pad images to the maximum size
    padded_images = []
    for img in images:
        h, w = img.shape[1:]
        # Calculate padding amounts
        pad_h = max_h - h
        pad_w = max_w - w
        
        # Pad the image. Assuming NCHW format
        padded_img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), 'constant', 0)
        padded_images.append(padded_img)
    
    # Stack padded images
    images_tensor = torch.stack(padded_images, dim=0)

    # Pad labels to the maximum number of boxes in the batch
    max_boxes = max(len(label) for label in labels)
    
    # Pad labels with zeros to a uniform size
    padded_labels = []
    for label in labels:
        num_boxes = len(label)
        # Create a zero tensor for padding
        padded_label = torch.zeros((max_boxes, 5), dtype=torch.float32)
        if num_boxes > 0:
            padded_label[:num_boxes, :] = label
        padded_labels.append(padded_label)

    # Stack padded labels
    labels_tensor = torch.stack(padded_labels, dim=0)

    return images_tensor, labels_tensor

