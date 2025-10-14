from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import numpy as np
import os
from torchvision.transforms import v2

# Add parent directory to path to allow imports from models/
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.YOLOBase import YOLOBase
from models.YOLOMax import YOLOMax
from helpers.loss import ComputeLoss
from helpers.data import YoloDataset, yolo_collate_fn
from utils.metrics import *
from utils.utils import *

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
    if not os.path.exists(os.path.join(run_dir, 'predictions')):
        os.makedirs(os.path.join(run_dir, 'predictions'))
    if not os.path.exists(os.path.join(run_dir, 'weights')):
        os.makedirs(os.path.join(run_dir, 'weights'))
    return run_dir
    
def load_weights(model, weights_path):
    """Load model weights from a .pt file."""
    if os.path.isfile(weights_path):
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        pretrained_weights = checkpoint['model'].float().state_dict()
        custom_model_dict = model.state_dict()
        name_map = {
            '0': 'b0', '1': 'b1', '2': 'b2', '3': 'b3', '4': 'b4', '5': 'b5',
            '6': 'b6', '7': 'b7', '8': 'b8', '9': 'h9', '12': 'h12', '15': 'h15',
            '16': 'h16', '18': 'h18', '19': 'h19', '21': 'h21', '22': 'h22',
            '25': 'h25', '28': 'h28', '29': 'detect'
        }
        mismatch = []
        match = []

        # Create a new state dict with the remapped names
        weights_to_load = {}
        for k, v in pretrained_weights.items():
            # Get the module index (e.g., '0', '1', '22')
            module_idx = k.split('.')[1]
            
            if module_idx in name_map:
                # Reconstruct the key with the new custom name
                custom_name = name_map[module_idx]
                rest_of_key = '.'.join(k.split('.')[2:])
                new_key = f"{custom_name}.{rest_of_key}"
                weights_to_load[new_key] = v
                match.append(new_key)
            else:
                mismatch.append(k)
        
        model.load_state_dict(weights_to_load, strict=False)
        print(f"Weights loaded from {weights_path}, totale: {len(match)} matched layers, {len(mismatch)} mismatched layers.")
    else:
        print(f"No weights file found at {weights_path}, training from scratch.")

# --- Main Training Function ---
def train(config, model, weights_path=None, cpus=4):
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
    
    # --- Data Augmentation ---
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    trtransforms = v2.Compose([
        v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
        # v2.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
        # v2.RandomApply([
        #     v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        #     v2.RandomPerspective(distortion_scale=0.2, p=0.5),
        # ], p=0.5),
        # v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        # v2.RandomErasing(p=.75, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
        v2.Normalize(mean=mean, std=std),
    ])

    valtransforms = v2.Compose([ 
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
        v2.Normalize(mean=mean, std=std),
    ])

    # --- DataLoaders ---
    train_img_dir = os.path.join(os.path.dirname(data_yaml_path), data_config['train'])
    train_img_dir = train_img_dir.replace('../', '')
    train_label_dir = train_img_dir.replace('images', 'labels')
    train_dataset = YoloDataset(img_dir=train_img_dir, label_dir=train_label_dir, img_size=img_size, transform=trtransforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=yolo_collate_fn, num_workers=cpus)

    val_img_dir = os.path.join(os.path.dirname(data_yaml_path), data_config['val'])
    val_img_dir = val_img_dir.replace('../', '')
    val_label_dir = val_img_dir.replace('images', 'labels')
    val_dataset = YoloDataset(img_dir=val_img_dir, label_dir=val_label_dir, img_size=img_size, transform=valtransforms)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=yolo_collate_fn, num_workers=cpus)

    # Model
    model = model(nc=nc)
    if weights_path:
        load_weights(model, weights_path)
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    compute_loss = ComputeLoss(model)
        
    # Training history
    history = {
        'train_loss': [], 
        'val_loss': [], 
        'train_box_loss': [], 
        'val_box_loss': [], 
        'train_cls_loss': [], 
        'val_cls_loss': [], 
        'train_dfl_loss': [], 
        'val_dfl_loss': [],
        'precision': [], 
        'recall': [], 
        'f1': [], 
        'map50': [], 
        'map95': [],
    }

    best_val_map = 0.0
    since_improved = 0

    # Training Loop
    for epoch in range(config['epochs']):
        model.train()
        train_loss = [0, 0, 0, 0]
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Training]")
        for images, targets in pbar_train:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss, components = compute_loss(preds, targets)
            loss.backward()
            optimizer.step()

            train_loss[0] += loss.item()
            train_loss[1] += components[0].item()
            train_loss[2] += components[1].item()
            train_loss[3] += components[2].item()

            pbar_train.set_postfix({
            'Loss': f"{loss.item() / batch_size:.4f}",
            'Box': f"{components[0]:.4f}",
            'Cls': f"{components[1]:.4f}",
            'Dfl': f"{components[2]:.4f}",
            'VRAM': f"{torch.cuda.memory_reserved()/1E9 if torch.cuda.is_available() else 0:.2f}GB"
        })
        
        avg_train_loss = train_loss[0] / (batch_size * len(train_loader))
        
        # Validation Loop
        model.eval()
        val_loss = [0, 0, 0, 0]

        # Reset metrics
        stats = []
        confusion_matrix = ConfusionMatrix(nc=nc)
        
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Validation]")
        with torch.no_grad():
            for images, targets in pbar_val:
                nb, _, height, width = images.shape
                images = images.to(device)
                targets = targets.to(device)
                
                preds, train_out = model(images)
                loss, components = compute_loss(train_out, targets)
                val_loss[0] += loss.item()
                val_loss[1] += components[0].item()
                val_loss[2] += components[1].item()
                val_loss[3] += components[2].item()

                targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
                lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)]  # for autolabelling
                
                preds = non_max_suppression(preds[0], labels=[], conf_thres=0.001 if epoch < 5 else 0.25, multi_label=True, agnostic=False)
                # Plot images
                if ((epoch+1) % 5 == 0 or epoch == 0) and len(stats) == 0:
                    log_random_image_predictions(
                        images.clone(), 
                        targets.clone(), 
                        preds.copy(), 
                        os.path.join(run_dir, f"predictions"), 
                        epoch, 
                        data_config['names']
                    )

                for si, pred in enumerate(preds):
                    labels = targets[targets[:, 0] == si, 1:]
                    nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
                    shape = (img_size, img_size)  # for scaling boxes
                    correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init

                    if npr == 0:
                        if nl:
                            stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                            confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                        continue

                    # Predictions
                    predn = pred.clone()
                    scale_boxes(images[si].shape[1:], predn[:, :4], shape)  # native-space pred

                    # Evaluate
                    if nl:
                        tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                        scale_boxes(images[si].shape[1:], tbox, shape)  # native-space labels
                        labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                        correct = process_batch(predn, labelsn, iouv)
                        confusion_matrix.process_batch(predn, labelsn)
                    stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)
                    
                pbar_val.set_postfix({
                    'Loss': f"{loss.item() / batch_size:.4f}",
                    'Box': f"{components[0]:.4f}",
                    'Cls': f"{components[1]:.4f}",
                    'Dfl': f"{components[2]:.4f}",
                    'VRAM': f"{torch.cuda.memory_reserved()/1E9 if torch.cuda.is_available() else 0:.2f}GB"
                })
        
        avg_val_loss = val_loss[0] / (batch_size * len(val_loader))
        
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
        print(f"          mAP50: {map50:.4f}, mAP95: {mean_ap:.4f}")
        print(f"          Precision: {mp:.4f}, Recall: {mr:.4f}, F1: {mf1:.4f}")


        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_box_loss'].append(train_loss[1] / len(train_loader))
        history['val_box_loss'].append(val_loss[1] / len(val_loader))
        history['train_cls_loss'].append(train_loss[2] / len(train_loader))
        history['val_cls_loss'].append(val_loss[2] / len(val_loader))
        history['train_dfl_loss'].append(train_loss[3] / len(train_loader))
        history['val_dfl_loss'].append(val_loss[3] / len(val_loader))
        history['precision'].append(mp)
        history['recall'].append(mr)
        history['f1'].append(mf1)
        history['map50'].append(map50)
        history['map95'].append(mean_ap)


        # Plot and save results after training is done
        plot_results(history, os.path.join(run_dir, 'results.png'))
        save_dict_to_csv(history, os.path.join(run_dir, 'results.csv'))
        confusion_matrix.plot(save_dir=run_dir, names=list(names.values()))

        # Save checkpoints
        last_ckpt_path = os.path.join(run_dir, 'weights', 'last.pt')
        torch.save(model.state_dict(), last_ckpt_path)

        if mean_ap > best_val_map:
            since_improved = 0
            best_val_map = mean_ap
            best_ckpt_path = os.path.join(run_dir, 'weights', f'best_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"New best model saved to {best_ckpt_path}")
        else:
            since_improved += 1

        if since_improved >= config.get('early_stopping', 20):
            print(f"No improvement for {since_improved} epochs. Early stopping.")
            break
    

if __name__ == '__main__':
    config = {
        'data_yaml': 'dataset/data.yaml',
        'img_size': 640,
        'batch_size': 16,
        'epochs': 500,
        'learning_rate': 0.001
    }
    train(config, YOLOMax, 'yolov9-t-converted.pt')

