import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import math
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision
from torchmetrics.detection import MeanAveragePrecision
import random
from torchvision.utils import draw_bounding_boxes, save_image

# Add parent directory to path to allow imports from models/
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.YOLOBase import YOLOBase
from helpers.data import YOLODataset

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
    
def xywh2xyxy(x):
    """Converts bounding box format from [center_x, center_y, width, height] to [x1, y1, x2, y2]."""
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def log_random_image_predictions(model, val_loader, device, run_dir, epoch, class_names):
    """Logs a random image with its ground truth and predicted bounding boxes."""
    model.eval()
    
    # Get a single batch from the validation loader for visualization
    try:
        images, targets = next(iter(val_loader))
    except StopIteration:
        print("Validation loader is empty, cannot log image.")
        return
        
    images = images.to(device)
    
    with torch.no_grad():
        inference_preds, _ = model(images)

    # Select a random image from the batch
    img_idx = random.randint(0, images.shape[0] - 1)
    img_tensor = images[img_idx]

    # Convert image tensor to uint8 for drawing
    img_to_draw = (img_tensor * 255).to(torch.uint8)
    img_h, img_w = img_tensor.shape[1:]

    # --- Get and Format Ground Truth Boxes (Green) ---
    gt_targets = targets[targets[:, 0] == img_idx, 1:]
    gt_boxes = gt_targets[:, 1:]
    # Denormalize
    gt_boxes_denorm = gt_boxes.clone()
    gt_boxes_denorm[:, 0] *= img_w
    gt_boxes_denorm[:, 1] *= img_h
    gt_boxes_denorm[:, 2] *= img_w
    gt_boxes_denorm[:, 3] *= img_h
    gt_boxes_xyxy = xywh2xyxy(gt_boxes_denorm)
    gt_labels = [class_names[int(c)] for c in gt_targets[:, 0]]

    # --- Get and Format Predicted Boxes (Blue) ---
    preds_for_img = inference_preds[img_idx]
    preds_for_img[:, 5:] *= preds_for_img[:, 4:5]  # conf = obj_conf * cls_conf
    
    vis_conf_thres = 0.25 
    conf, labels_idx = preds_for_img[:, 5:].max(1)
    
    keep_indices = conf > vis_conf_thres
    
    pred_boxes = preds_for_img[keep_indices, :4]
    pred_boxes_xyxy = xywh2xyxy(pred_boxes)
    pred_scores = conf[keep_indices]
    pred_labels_idx = labels_idx[keep_indices]
    
    pred_labels = [f"{class_names[int(l)]} {s:.2f}" for l, s in zip(pred_labels_idx, pred_scores)]

    # Draw boxes on the image
    # Draw GT first, then predictions on the result
    if gt_boxes_xyxy.shape[0] > 0:
        img_to_draw = draw_bounding_boxes(img_to_draw.cpu(), boxes=gt_boxes_xyxy, labels=gt_labels, colors="green", width=2)
    if pred_boxes_xyxy.shape[0] > 0:
        img_to_draw = draw_bounding_boxes(img_to_draw, boxes=pred_boxes_xyxy, labels=pred_labels, colors="blue", width=2)

    # Save the image
    save_path = os.path.join(run_dir, f"epoch_{epoch+1}_predictions.jpg")
    save_image(img_to_draw / 255.0, save_path)
    
# --- Loss and Metrics Calculation ---
def lossfn(predictions, targets, num_classes, lambda_coord=5, lambda_noobj=0.5):
    """
    Computes YOLO loss.
    - predictions: Predicted tensor.
    - targets: Ground truth tensor.
    """
    # Unpack predictions and targets
    pred_boxes = predictions[..., :4]
    pred_conf = predictions[..., 4]
    pred_classes = predictions[..., 5:]
    target_boxes = targets[..., :4]
    target_conf = targets[..., 4]
    target_classes = targets[..., 5:]
    
    # Localization Loss
    box_loss = lambda_coord * torch.sum((pred_boxes - target_boxes) ** 2)

    # Confidence Loss
    obj_loss = torch.sum((pred_conf - target_conf) ** 2)
    noobj_loss = lambda_noobj * torch.sum((pred_conf[target_conf == 0]) ** 2)

    # Classification Loss
    class_loss = torch.sum((pred_classes - target_classes) ** 2)

    # Total Loss
    total_loss = box_loss + obj_loss + noobj_loss + class_loss
    return total_loss

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
        
    nc = len(data_config['names'])
    
    # -- Transforms ---
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    trtransforms = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])
    valtransforms = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])

    # --- DataLoaders ---
    train_img_dir = os.path.join(os.path.dirname(data_yaml_path), data_config['train'])
    train_img_dir = train_img_dir.replace('../', '')
    train_label_dir = train_img_dir.replace('images', 'labels')
    train_dataset = YOLODataset(img_dir=train_img_dir, label_dir=train_label_dir, transforms=trtransforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    val_img_dir = os.path.join(os.path.dirname(data_yaml_path), data_config['val'])
    val_img_dir = val_img_dir.replace('../', '')
    val_label_dir = val_img_dir.replace('images', 'labels')
    val_dataset = YOLODataset(img_dir=val_img_dir, label_dir=val_label_dir, transforms=valtransforms)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model
    model = YOLOBase(num_classes=nc, num_anchors=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    compute_loss = lossfn
    
    # --- TorchMetrics ---
    metric = MeanAveragePrecision(box_format='xywh', backend="faster_coco_eval").to(device)
    
    # Training history
    history = {
        'train_loss': [], 
        'val_loss': [], 
        'map_0.5': [],
        'map_0.5:0.95': [],
        'mar_100': []
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
            
            preds = model(images)
            loss, components = compute_loss(preds, targets, num_classes=nc)
            
            # Back propagation
            optimizer.zero_grad()
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

        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Validation]")
        with torch.no_grad():
            for images, targets in pbar_val:
                images = images.to(device)
                targets = targets.to(device)
                
                inference_preds, train_preds = model(images)
                loss, _ = compute_loss(train_preds, targets)
                val_loss += loss.item()
                
                # Format for TorchMetrics
                preds_for_metric = []
                for pred in inference_preds:
                    pred[:, 5:] *= pred[:, 4:5]  # conf = obj_conf * cls_conf
                    conf, _ = pred[:, 5:].max(1)
                    
                    
                    if pred.shape[0] > 100:
                        # Get the indices of the top K predictions by confidence
                        topk_indices = conf.topk(100, largest=True).indices
                        pred = pred[topk_indices]
                        
                    conf, labels = pred[:, 5:].max(1)
                    
                    preds_for_metric.append(dict(
                        boxes=pred[:, :4],
                        scores=conf,
                        labels=labels.int()
                    ))
                    
                targets_for_metric = []
                # Un-normalize and format ground truth
                scaled_targets = targets.clone()
                scaled_targets[:, 2:] *= torch.tensor([images.shape[3], images.shape[2], images.shape[3], images.shape[2]], device=device)
                for i in range(images.shape[0]):
                    labels = scaled_targets[scaled_targets[:, 0] == i, 1:]
                    targets_for_metric.append(dict(
                        boxes=labels[:, 1:],
                        labels=labels[:, 0].int()
                    ))
                                
                metric.update(preds_for_metric, targets_for_metric)

        avg_val_loss = val_loss / len(val_loader)
        
        # Compute and log metrics
        results = metric.compute()
        # metric.reset()

        map50 = results['map_50'].item()
        map95 = results['map'].item()
        mar100 = results['mar_100'].item()

        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, mAP@.50: {map50:.4f}, mAP@.50-.95: {map95:.4f}, mAR@100: {mar100:.4f}")

        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['map_0.5'].append(map50)
        history['map_0.5:0.95'].append(map95)
        history['mar_100'].append(mar100)
        
        # Log random image predictions
        log_random_image_predictions(model, val_loader, device, run_dir, epoch, data_config['names'])

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
        'batch_size': 128,
        'epochs': 10,
        'learning_rate': 0.001
    }
    train(config)

