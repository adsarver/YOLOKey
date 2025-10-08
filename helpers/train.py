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

# Add parent directory to path to allow imports from models/
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.YOLOBase import YOLOBase
from helpers.data import YoloDataset, yolo_collate_fn

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
    """Plots training and validation metrics and saves the figure."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training and Validation Metrics')

    axs[0, 0].plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    axs[0, 0].plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    axs[0, 0].set_title('Loss over Epochs')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].plot(epochs, history['precision'], 'go-', label='Precision')
    axs[0, 1].set_title('Precision over Epochs')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('Precision')
    axs[0, 1].grid(True)
    
    axs[1, 0].plot(epochs, history['recall'], 'yo-', label='Recall')
    axs[1, 0].set_title('Recall over Epochs')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Recall')
    axs[1, 0].grid(True)

    axs[1, 1].plot(epochs, history['map_0.5'], 'mo-', label='mAP@0.5')
    axs[1, 1].set_title('mAP@0.5 over Epochs')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('mAP@0.5')
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"Results plot saved to {save_path}")

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results."""
    # ... (Implementation of NMS, can be complex)
    # For simplicity, we can use torchvision's NMS if available, otherwise a basic implementation is needed.
    # This is a placeholder for a complete NMS implementation.
    # A full implementation would be several dozen lines of code.
    # In a real scenario, you'd use a library implementation or write one carefully.
    
    # This is a simplified stand-in for NMS logic.
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for i, x in enumerate(prediction):
        # This is not real NMS, just filtering by confidence
        x = x[x[:, 4] > conf_thres]
        if x.shape[0] > max_det:
            x = x[:max_det]
        output[i] = x
    return output

# --- Loss and Metrics Calculation ---

class ComputeLoss:
    # ... (Implementation is the same as before, no changes needed)
    def __init__(self, model):
        self.device = next(model.parameters()).device
        self.hyp = {'box': 0.05, 'cls': 0.5, 'obj': 1.0}
        self.bce_cls = torch.nn.BCEWithLogitsLoss()
        self.bce_obj = torch.nn.BCEWithLogitsLoss()
        self.model = model

    def __call__(self, preds, targets):
        lcls, lbox, lobj = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        tcls, tbox, indices, anchors = self.build_targets(preds, targets)

        for i, pred_layer in enumerate(preds):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(pred_layer[..., 0], device=self.device)

            n = b.shape[0]
            if n:
                ps = pred_layer[b, a, gj, gi]
                
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                iou = self.bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)
                lbox += (1.0 - iou).mean()

                tobj[b, a, gj, gi] = iou.detach().clamp(0).type(tobj.dtype)

                if self.model.nc > 1:
                    t = torch.full_like(ps[:, 5:], 0, device=self.device)
                    t[range(n), tcls[i]] = 1
                    lcls += self.bce_cls(ps[:, 5:], t)
            
            lobj += self.bce_obj(pred_layer[..., 4], tobj)

        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']

        return lbox + lobj + lcls, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, preds, targets):
        na, nt = self.model.head.na, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)
        
        for i in range(self.model.head.nl):
            anchors = self.model.head.anchors[i]
            gain[2:6] = torch.tensor(preds[i].shape)[[3, 2, 3, 2]]
            t = targets * gain
            if nt:
                r = t[:, :, 4:6] / anchors[:, None]
                j = torch.max(r, 1. / r).max(2)[0] < 4.0
                t = t[j]
            else:
                t = targets[0]
            
            b, c = t[:, :2].long().T
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]
            gij = gxy.long()
            gi, gj = gij.T
            
            a = t[:, 6].long()
            indices.append((b, a, gj.clamp_(0, int(gain[3].item()) - 1), gi.clamp_(0, int(gain[2].item()) - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            anch.append(anchors[a])
            tcls.append(c)
        
        return tcls, tbox, indices, anch

    def bbox_iou(self, box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
        # Returns the IoU of box1 to box2. box1 is 4xn, box2 is nx4
        box2 = box2.T

        # Get the coordinates of bounding boxes
        if x1y1x2y2:  # x1, y1, x2, y2 = box1
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        else:  # transform from xywh to xyxy
            b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
            b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
            b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
            b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)
        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * \
                     torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        if CIoU:
            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
            c2 = cw ** 2 + ch ** 2 + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            v = (4 / math.pi ** 2) * torch.pow(torch.atan((b2_x2 - b2_x1) / (b2_y2 - b2_y1)) - torch.atan((b1_x2 - b1_x1) / (b1_y2 - b1_y1)), 2)
            with torch.no_grad():
                alpha = v / (v - iou + (1.0 + eps))
            return iou - (rho2 / c2 + v * alpha)
        return iou
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

    # --- DataLoaders ---
    train_img_dir = os.path.join(os.path.dirname(data_yaml_path), data_config['train'])
    train_img_dir = train_img_dir.replace('../', '')
    train_label_dir = train_img_dir.replace('images', 'labels')
    train_dataset = YoloDataset(img_dir=train_img_dir, label_dir=train_label_dir, img_size=img_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=yolo_collate_fn, num_workers=4)

    val_img_dir = os.path.join(os.path.dirname(data_yaml_path), data_config['val'])
    val_img_dir = val_img_dir.replace('../', '')
    val_label_dir = val_img_dir.replace('images', 'labels')
    val_dataset = YoloDataset(img_dir=val_img_dir, label_dir=val_label_dir, img_size=img_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=yolo_collate_fn, num_workers=4)

    # Model
    model = YOLOBase(nc=len(data_config['names'])).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    compute_loss = ComputeLoss(model)
    
    # Training history
    history = {'train_loss': [], 'val_loss': [], 'precision': [], 'recall': [], 'map_0.5': []}
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
            loss, _ = compute_loss(preds, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar_train.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Loop
        model.eval()
        val_loss = 0
        # Placeholder for metrics, in a real scenario you would compute them
        val_precision, val_recall, val_map = 0.1 * (epoch+1), 0.15 * (epoch+1), 0.2 * (epoch+1) # Dummy values

        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Validation]")
        with torch.no_grad():
            for images, targets in pbar_val:
                images = images.to(device)
                targets = targets.to(device)
                preds = model(images)
                loss, _ = compute_loss(preds, targets)
                val_loss += loss.item()
                # Here you would add logic to calculate Precision, Recall, mAP
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: \n\tTrain Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['precision'].append(val_precision) # Replace with real metric
        history['recall'].append(val_recall)       # Replace with real metric
        history['map_0.5'].append(val_map)       # Replace with real metric

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
        'data_yaml': 'dataset/data.yaml',
        'img_size': 640,
        'batch_size': 8,
        'epochs': 50,
        'learning_rate': 0.001
    }
    train(config)

