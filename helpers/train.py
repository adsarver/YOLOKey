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
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('Training and Validation Metrics')

    axs[0, 0].plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    axs[0, 0].plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    axs[0, 0].set_title('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].plot(epochs, history['precision'], 'go-', label='Precision')
    axs[0, 1].set_title('Precision')
    axs[0, 1].grid(True)
    
    axs[0, 2].plot(epochs, history['recall'], 'yo-', label='Recall')
    axs[0, 2].set_title('Recall')
    axs[0, 2].grid(True)

    axs[1, 0].plot(epochs, history['map_0.5'], 'mo-', label='mAP@0.5')
    axs[1, 0].set_title('mAP@0.5')
    axs[1, 0].grid(True)
    
    axs[1, 1].plot(epochs, history['map_0.5:0.95'], 'co-', label='mAP@.5:.95')
    axs[1, 1].set_title('mAP@.5-.95')
    axs[1, 1].grid(True)
    
    axs[1, 2].plot(epochs, history['f1'], 'ko-', label='F1 Score')
    axs[1, 2].set_title('F1 Score')
    axs[1, 2].grid(True)
    
    axs[2, 1].plot(epochs, history['batch_iou'], 'ro-', label='Avg Max IoU')
    axs[2, 1].set_title('Avg Max IoU')
    axs[2, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"Results plot saved to {save_path}")
    
def xywh2xyxy(x):
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=300):
    """Performs Non-Maximum Suppression (NMS) on inference results."""
    bs = prediction.shape[0]
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    
    for xi, x in enumerate(prediction):
        # Calculate the final confidence score
        x[:, 5:] *= x[:, 4:5] # conf = obj_conf * cls_conf
        
        # Get boxes and scores
        box = xywh2xyxy(x[:, :4])
        conf, j = x[:, 5:].max(1, keepdim=True)
        
        # Filter based on final confidence
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        
        if not x.shape[0]:
            continue

        # Perform NMS
        boxes, scores = x[:, :4], x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        output[xi] = x[i]

    return output

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves. """
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    unique_classes = np.unique(target_cls)
    ap, p, r = np.zeros((len(unique_classes), tp.shape[1])), np.zeros((len(unique_classes), 1000)), np.zeros((len(unique_classes), 1000))

    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()
        n_p = i.sum()

        if n_p == 0 or n_l == 0:
            continue
        
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        recall_curve = tpc / (n_l + 1e-16)
        r[ci] = np.interp(-conf[i], -conf[i], recall_curve, left=0)

        precision_curve = tpc / (tpc + fpc)
        p[ci] = np.interp(-conf[i], -conf[i], precision_curve, left=1)

        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(r[ci], p[ci])

    f1 = 2 * p * r / (p + r + 1e-16)
    return p, r, ap, f1, unique_classes.astype('int32')

def compute_ap(recall, precision):
    """ Compute the average precision from the recall and precision curves. """
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap, mpre, mrec

def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = torchvision.ops.box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.from_numpy(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct

# --- Loss and Metrics Calculation ---

class ComputeLoss:
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
    history = {
        'train_loss': [], 
        'val_loss': [], 
        'f1': [],
        'precision': [], 
        'recall': [], 
        'map_0.5': [],
        'map_0.5:0.95': [],
        'batch_iou': [],
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
        stats = []
        iouv = torch.linspace(0.5, 0.95, 10).to(device)
        val_loss = 0

        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Validation]")
        with torch.no_grad():
            batch_max_ious = []
            for images, targets in pbar_val:
                images = images.to(device)
                targets = targets.to(device)
                targets[:, 2:] *= torch.Tensor([images.shape[3], images.shape[2], images.shape[3], images.shape[2]]).to(device)
                
                inference_preds, train_preds = model(images)
                loss, _ = compute_loss(train_preds, targets)
                val_loss += loss.item()

                inference_preds = non_max_suppression(inference_preds, conf_thres=0.001, iou_thres=0.6)
                
                for i, pred in enumerate(inference_preds):
                    labels = targets[targets[:, 0] == i, 1:]
                    nl = len(labels)
                    tcls = labels[:, 0].tolist() if nl else []
                    
                    if len(pred) > 0 and nl > 0:
                        ious = torchvision.ops.box_iou(xywh2xyxy(labels[:, 1:]), pred[:, :4])
                        if ious.numel() > 0:
                            batch_max_ious.append(ious.max().item())
                            
                    if len(pred) == 0:
                        if nl:
                            stats.append((torch.zeros(0, iouv.shape[0], dtype=torch.bool), torch.zeros(0), torch.zeros(0), torch.tensor(tcls)))
                        continue
                    
                    labels_xyxy = labels.clone()
                    labels_xyxy[:, 1:] = xywh2xyxy(labels[:, 1:])
                    
                    correct = process_batch(pred, labels_xyxy, iouv)
                    stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), torch.tensor(tcls)))
                    avg_max_iou = np.mean(batch_max_ious) if batch_max_ious else 0.0
                    pbar_val.set_postfix({
                        'loss': f'{loss.item():.4f}', 
                        'avg_max_iou': f'{avg_max_iou:.3f}', 
                        'VRAM': f"{torch.cuda.memory_reserved()/1E9 if torch.cuda.is_available() else 0:.2f}GB"
                    })
        
        avg_val_loss = val_loss / len(val_loader)
        
        stats = [np.concatenate(x, 0) for x in zip(*stats)]
        
        if len(stats) and stats[0].any():
            p, r, ap, f1, _ = ap_per_class(*stats)
            ap50, ap = ap[:, 0], ap.mean(1)
            mp, mr, map50, map95, f1 = p.mean(), r.mean(), ap50.mean(), ap, f1.mean()
        else:
            mp, mr, map50, map95, f1 = 0., 0., 0., 0., 0.

        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, P: {mp:.4f}, R: {mr:.4f}, mAP@.5: {map50:.4f}, mAP@.5-.95: {map95:.4f}, F1: {f1:.4f}")

        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['f1'].append(f1)
        history['precision'].append(mp)
        history['recall'].append(mr)
        history['map_0.5'].append(map50)
        history['map_0.5:0.95'].append(map95)
        history['batch_iou'].append(np.mean(batch_max_ious) if batch_max_ious else 0.0)

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
        'batch_size': 16,
        'epochs': 50,
        'learning_rate': 0.001
    }
    train(config)

