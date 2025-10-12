import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import yaml
from glob import glob

class YoloDataset(Dataset):
    """
    YOLOv9 Dataset Loader.
    This class loads images and their corresponding labels from a directory structure
    that follows the Ultralytics YOLO format.
    """
    def __init__(self, img_dir, label_dir, img_size=640, transform=None):
        """
        Args:
            img_dir (str): Path to the directory containing images.
            label_dir (str): Path to the directory containing label files.
            img_size (int): The target size to resize images to.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.transform = transform

        # Find all image files and filter out any that don't have corresponding labels
        self.image_files = sorted(glob(os.path.join(self.img_dir, '*.*')))
        self.image_files = [img for img in self.image_files if os.path.exists(self._get_label_path(img))]


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]

        # --- Load Image ---
        # Using OpenCV to load the image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- Load Labels ---
        label_path = self._get_label_path(img_path)
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()

                    # Bbox format: [class_id, x_center, y_center, width, height]
                    bbox = [float(p) for p in parts[:5]]
                    
                    labels.append(bbox)
        labels = torch.tensor(labels, dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        return img, labels

    def _get_label_path(self, img_path):
        """Helper to find the corresponding label file path for an image."""
        filename = os.path.splitext(os.path.basename(img_path))[0]
        return os.path.join(self.label_dir, f"{filename}.txt")

def yolo_collate_fn(batch):
    """
    Custom collate function for YOLO datasets.
    It handles images and labels of varying quantities.
    """
    images, labels = zip(*batch)
    
    # Stack images into a single tensor
    images = torch.stack(images, 0)
    
    # Convert labels tuple to a list to allow item assignment
    labels = list(labels)

    # Add a batch index to each label tensor and concatenate
    for i, label in enumerate(labels):
        if len(label) > 0:
            # Add batch index as the first column
            label = torch.cat([torch.full((label.shape[0], 1), i), label], dim=1)
        # Handle images with no labels
        else:
            label = torch.empty(0, 6) # batch_idx, class_id, x, y, w, h
        labels[i] = label

    return images, torch.cat(labels, 0)


# --- Example Usage ---
if __name__ == '__main__':
    # Parse the data.yaml file to get dataset paths and info
    data_yaml_path = 'dataset/data.yaml'
    try:
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)

        # Construct absolute paths
        train_img_dir = os.path.join('dataset', data_config['train'][2:])
        train_label_dir = train_img_dir.replace('images', 'labels')
        
        val_img_dir = os.path.join('dataset', data_config['val'][2:])
        val_label_dir = val_img_dir.replace('images', 'labels')
        
        num_classes = data_config['nc']
        class_names = data_config['names']

        print(f"Dataset found. Number of classes: {num_classes}")
        print(f"Class names: {class_names}")
        print(f"Train images path: {train_img_dir}")
        print(f"Validation images path: {val_img_dir}")

        # Create dataset and dataloader
        train_dataset = YoloDataset(img_dir="dataset/train/images", label_dir="dataset/train/labels")
        
        # Check if the dataset is empty
        if len(train_dataset) == 0:
             print("\nError: No images found in the training directory or no matching labels.")
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=4,
                shuffle=True,
                collate_fn=yolo_collate_fn
            )

            # Get one batch and inspect it
            print(f"\nTesting the DataLoader with a batch size of 4...")
            images, labels = next(iter(train_loader))

            print(f"\nShape of images tensor: {images.shape}")
            print(f"Shape of labels tensor: {labels.shape}")
            print(f"First 5 labels:\n {labels[:5]}")
            print("\nLabels format: [batch_index, class_id, x_center, y_center, width, height]")


    except FileNotFoundError:
        print(f"Error: {data_yaml_path} not found.")
        print("Please ensure the data.yaml file is in the 'dataset' directory.")
    except Exception as e:
        print(f"An error occurred: {e}")
