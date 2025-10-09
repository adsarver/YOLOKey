from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import math
import numpy as np
import os
import matplotlib.pyplot as plt
from torchmetrics.detection import MeanAveragePrecision
import random
from torchvision.utils import draw_bounding_boxes, save_image
from torchvision.transforms import v2
import cv2

# Add parent directory to path to allow imports from models/
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.YOLOBase import YOLOBase
from helpers.loss import ComputeLoss
from helpers.data import YoloDataset, yolo_collate_fn
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou, non_max_suppression, output_to_target, process_batch, scale_boxes
from utils.utils import xywh2xyxy, colors

# --- Utility Functions ---

def get_next_run_dir(base_dir='runs'):
    """Gets the next available run directory."""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    existing_runs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('run_')]
    if not existing_runs:
        run_dir = os.path.join(base_dir, 'run_1')
    else:
        max_run_num = 0
        for run in existing_runs:
            try:
                run_num = int(run.split('_')[1])
                if run_num > max_run_num:
                    max_run_num = run_num
            except (IndexError, ValueError):
                continue
        run_dir = os.path.join(base_dir, f'run_{max_run_num + 1}')
    os.makedirs(run_dir)
    return run_dir

def plot_results(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axs = plt.subplots(2, 2, figsize=(15, 10)) # Adjusted for 4 plots
    fig.suptitle('Training and Validation Metrics')

    axs[0, 0].plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    axs[0, 0].plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    axs[0, 0].set_title('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].plot(epochs, history['map_0.5'], 'mo-', label='mAP@.50')
    axs[0, 1].set_title('mAP@.50')
    axs[0, 1].grid(True)
    
    axs[1, 0].plot(epochs, history['map_0.5:0.95'], 'co-', label='mAP@.50:.95')
    axs[1, 0].set_title('mAP@.50-.95 (Primary Metric)')
    axs[1, 0].grid(True)
    
    axs[1, 1].plot(epochs, history['mar_100'], 'yo-', label='mAR@100')
    axs[1, 1].set_title('Mean Average Recall @ 100 Detections')
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"Results plot saved to {save_path}")

# --- Main Training Function ---
def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create run directory
    run_dir = get_next_run_dir()
    print(f"Results will be saved in: {run_dir}")

    # --- Load Configuration ---
    data_yaml_path = config['data_yaml']
    batch_size = config['batch_size']
    img_size = config['img_size']
    
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
        
    # --- Initialize Metrics ---
    names = data_config['names']
    nc = len(names)
    names = dict(enumerate(names))
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # IoU vector for mAP@0.5:0.95
    niou = iouv.numel()
    
    trtransforms = v2.Compose([
        v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
        v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.2),
        v2.RandomErasing(scale=(0.02, 0.33), ratio=(0.3, 3.3), value=1, inplace=False),
        v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    valtransforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
        # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- DataLoaders ---
    train_img_dir = os.path.join(os.path.dirname(data_yaml_path), data_config['train'])
    train_img_dir = train_img_dir.replace('../', '')
    train_label_dir = train_img_dir.replace('images', 'labels')
    train_dataset = YoloDataset(img_dir=train_img_dir, label_dir=train_label_dir, img_size=img_size, transform=trtransforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=yolo_collate_fn, num_workers=4)

    val_img_dir = os.path.join(os.path.dirname(data_yaml_path), data_config['val'])
    val_img_dir = val_img_dir.replace('../', '')
    val_label_dir = val_img_dir.replace('images', 'labels')
    val_dataset = YoloDataset(img_dir=val_img_dir, label_dir=val_label_dir, img_size=img_size, transform=valtransforms)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=yolo_collate_fn, num_workers=4)

    # Model
    model = YOLOBase(nc=nc).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    compute_loss = ComputeLoss(model)
    
    # --- TorchMetrics ---
    metric = MeanAveragePrecision(box_format='xywh', backend="faster_coco_eval").to(device)
    
    # Training history
    history = {
        'train_loss': [], 
        'val_loss': [], 
        'precision': [], 
        'recall': [], 
        'f1': [], 
        'map_0.5': [], 
        'map_0.5:0.95': [],
    }

    best_val_loss = float('inf')

    # Training Loop
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Training]")
        for images, targets in pbar_train:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss, components = compute_loss(preds, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            pbar_train.set_postfix({
            'Loss': f"{loss.item():.4f}",
            'Box': f"{components[0]:.4f}",
            'Obj': f"{components[1]:.4f}",
            'Cls': f"{components[2]:.4f}",
            'VRAM': f"{torch.cuda.memory_reserved()/1E9 if torch.cuda.is_available() else 0:.2f}GB"
        })
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Loop
        model.eval()
        val_loss = 0

        # Reset metrics
        stats = []
        confusion_matrix = ConfusionMatrix(nc=nc)
        
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Validation]")
        with torch.no_grad():
            for images, targets in pbar_val:
                images = images.to(device)
                targets = targets.to(device)
                
                preds = model(images)
                loss, components = compute_loss(preds, targets)
                val_loss += loss.item()
                
                preds = non_max_suppression(preds[0], conf_thres=0.001, iou_thres=0.7, labels=data_config.get('labels'), multi_label=True, agnostic=False, max_det=100)
                for si, pred in enumerate(preds):
                    labels = torch.Tensor(targets[targets[:, 0] == si, 1:]).to(device)  # labels for image si
                    nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
                    # path, shape = Path(paths[si]), shapes[si][0]
                    correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init

                    if npr == 0:
                        if nl:
                            stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                            confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                        continue

                    # Predictions
                    predn = pred.clone()

                    # Evaluate
                    if nl:
                        tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                        labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                        correct = process_batch(predn, labelsn, iouv)
                        confusion_matrix.process_batch(predn, labelsn)
                    stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

                # Plot images
                # plot_images(images, targets, paths, run_dir / f'val_batch{batch_i}_labels.jpg', names)  # labels
                # plot_images(images, output_to_target(preds), paths, run_dir / f'val_batch{batch_i}_pred.jpg', names)  # pred

                pbar_val.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Box': f"{components[0]:.4f}",
                    'Obj': f"{components[1]:.4f}",
                    'Cls': f"{components[2]:.4f}",
                    'VRAM': f"{torch.cuda.memory_reserved()/1E9 if torch.cuda.is_available() else 0:.2f}GB"
                })                       
                                
        avg_val_loss = val_loss / len(val_loader)
        
        # Compute metrics
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)] # Compile stats
        if len(stats) and len([x for x in stats if x.any()]):
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=True, save_dir=run_dir, names=names)
            ap50, ap = ap[:, 0], ap.mean(1) # AP@0.5, AP@0.5:0.95
            mf1, mp, mr, map50, mean_ap = f1.mean(), p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(int), minlength=nc) # Number of targets per class
        else:
            nt = torch.zeros(1)
            mf1, mp, mr, map50, mean_ap = 0.0, 0.0, 0.0, 0.0, 0.0
        
        # Compute and log metrics
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"          mAP@.5: {map50:.4f}, mAP@.5:.95: {mean_ap:.4f}, Precision: {mp:.4f}, Recall: {mr:.4f}")

        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['precision'].append(mp)
        history['recall'].append(mr)
        history['f1'].append(mf1)
        history['map_0.5'].append(map50)
        history['map_0.5:0.95'].append(mean_ap)

        # Save checkpoints
        last_ckpt_path = os.path.join(run_dir, 'last.pt')
        torch.save(model.state_dict(), last_ckpt_path)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_ckpt_path = os.path.join(run_dir, 'best.pt')
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"New best model saved to {best_ckpt_path}")

    # Plot and save results after training is done
    plot_results(history, os.path.join(run_dir, 'results.png'))

if __name__ == '__main__':
    config = {
        'data_yaml': 'dataset128/data.yaml',
        'img_size': 128,
        'batch_size': 64,
        'epochs': 500,
        'learning_rate': 0.001
    }
    train(config)

