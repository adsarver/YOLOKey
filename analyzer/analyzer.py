import matplotlib.pyplot as plt
import cv2
import os
import torch
from torchvision.transforms import v2
import torch.nn.functional as F
import torch.nn as nn
from utils.model_utils import *
from utils.utils import *
from utils.metrics import non_max_suppression
from models.YOLOBase import YOLOBase
from models.YOLOMax import YOLOMax
from models.YOLODrop import YOLODrop
from models.YOLOCBAM import YOLOCBAM
from models.YOLOP2 import YOLOP2
from models.YOLOHD import YOLOHD

FILE_OUTPUT = 'output/'
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'Comma', 'D', 'DOT', 'E', 'ENTER', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'MINUS', 'N', 'O', 'P', 'PLUS', 'Q', 'R', 'S', 'SPACE', 'T', 'TAB', 'U', 'V', 'W', 'X', 'Y', 'Z', 'accent', 'keyboard']

def create_directory():
    os.makedirs(FILE_OUTPUT, exist_ok=True)
    
    for folder in os.listdir(FILE_OUTPUT):
        folder_path = os.path.join(FILE_OUTPUT, folder)
        # Get index of last run
        if folder.startswith('output') and os.path.isdir(folder_path):
            try:
                run_index = int(folder.split('output')[-1])
                if run_index >= new_run_index:
                    new_run_index = run_index + 1

            except ValueError:
                continue

    os.makedirs(f'{FILE_OUTPUT}output{new_run_index}', exist_ok=True)
    

def save_image(image, name):
    cv2.imwrite(f'{name}.jpg', image)

def show_plot(data, metric_name, offsety=-10):
    plt.title(f'{metric_name} over Epochs')
    for name, df in data.items():
        offsety -= 20
        best_epoch = df[metric_name].idxmax() + 1
        plt.plot(df[metric_name], label=name)
        plt.annotate(
            f'{name} at {df[metric_name].max() * 100.0:.2f}% @ E{best_epoch}',
            (best_epoch-1, df[metric_name].max()),
            textcoords="offset points",
            xytext=(0,offsety),
            ha='center',
            arrowprops=dict(arrowstyle="->", color='red')
        )
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()
    
def ablation_study(data):
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.axis('tight')
    ax.set_title('Ablation Study - mAP@0.95 Comparison')
    columns = [x.replace('_', ' ').title() for x in data["YOLOBase"].columns if "loss"][8:]
    columns.insert(0, 'Model')
    columns.append('Best Epoch')
    data_table = []

    for name, df in data.items():
        best_epoch = df['map95'].idxmax() + 1
        row = [f'{x*100.0:.2f}' for x in df.loc[best_epoch]][8:]
        data_table.append([name] + row + [best_epoch])
    data_table.sort(key=lambda x: float(x[-2]), reverse=True)
    table = ax.table(cellText=data_table, colLabels=columns, loc='center')
    plt.show()


def import_data(file_path):
    data = {}
    models = {
        'YOLOBase': YOLOBase,
        'YOLOMax': YOLOMax,
        'YOLODrop': YOLODrop,
        # 'YOLOCBAM': YOLOCBAM,
        # 'YOLOP2': YOLOP2,
        # 'YOLOHD': YOLOHD
    }
    
    for folder in os.listdir(file_path):
        folder_path = os.path.join(file_path, folder, 'results.csv')
        df = pd.read_csv(folder_path, sep=",")
        data[folder] = df
        model_path = ''
        max_epoch = 0

        for model_file in os.listdir(os.path.join(file_path, folder, 'weights')):
            if model_file.startswith('best_epoch') and int(model_file.split('_')[-1].split('.')[0]) >= max_epoch:
                max_epoch = int(model_file.split('_')[-1].split('.')[0])
                model_path = os.path.join(file_path, folder, 'weights', model_file)
        if folder in models:
            print(max_epoch)
            print(model_path)
            model_class = models[folder]
            model_instance = model_class(nc=len(class_names))
            state_dict = torch.load(model_path)
            model_instance.load_state_dict(state_dict)
            models[folder] = model_instance
            
    return data, models


def run_detect(model, image_path, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), scale_factor=3):
    font = None
    import matplotlib.font_manager as fm
    for font in fm.findSystemFonts(fontpaths=['/usr/share/fonts/truetype/']):
        if 'Serif-Regular' in font:
            font = font
            break
        
    model.to(device)
    model.eval()
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    transforms = v2.Compose([ 
        v2.ToImage(),
        v2.Resize((640, 640)),
        v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
        v2.Normalize(mean=mean, std=std),
    ])
    
    img = transforms(img).to(device)
    img_h, img_w = img.shape[1:]
    
    img_model = img.unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        pred = model(img_model)[0]
        pred = non_max_suppression(pred, 0.50, 0.45, multi_label=True, agnostic=False)
    
    img = un_normalize_image(img)
    high_res_img = F.interpolate(
        img.unsqueeze(0).float(),
        size=(img_h * scale_factor, img_w * scale_factor),
        mode='bilinear', 
        align_corners=False
    ).squeeze(0).to(torch.uint8)
    
    # Convert image tensor to uint8 for drawing
    img_h, img_w = high_res_img.shape[1:]
    
    # --- Get and Format Predicted Boxes (Blue) ---
    preds_for_img = pred[0]
    conf = preds_for_img[:, 4]
    labels_idx = preds_for_img[:, 5]
    
    pred_boxes = preds_for_img[:, :4] * scale_factor
    pred_scores = conf
    pred_labels_idx = labels_idx
    pred_labels = [f"{class_names[int(l)]} {s:.2f}" for l, s in zip(pred_labels_idx, pred_scores)]

    img_to_draw = draw_bounding_boxes(
        high_res_img, 
        boxes=pred_boxes, 
        labels=pred_labels, 
        colors="red", 
        width=2*scale_factor, 
        font_size=10*scale_factor, 
        font=font
    )

    return img_to_draw.cpu().numpy().transpose(1, 2, 0)

def detect_stream(model, img, transforms, font, scale_factor=3):   
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = transforms(img).to(device)
    img_h, img_w = img.shape[1:]
    
    img_model = img.unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        pred = model(img_model)[0]
        pred = non_max_suppression(pred, 0.50, 0.45, multi_label=True, agnostic=False)
    
    img = un_normalize_image(img)
    high_res_img = F.interpolate(
        img.unsqueeze(0).float(),
        size=(img_h * scale_factor, img_w * scale_factor),
        mode='bilinear', 
        align_corners=False
    ).squeeze(0).to(torch.uint8)
    
    # Convert image tensor to uint8 for drawing
    img_h, img_w = high_res_img.shape[1:]
    
    # --- Get and Format Predicted Boxes (Blue) ---
    preds_for_img = pred[0]
    conf = preds_for_img[:, 4]
    labels_idx = preds_for_img[:, 5]
    
    pred_boxes = preds_for_img[:, :4] * scale_factor
    pred_scores = conf
    pred_labels_idx = labels_idx
    pred_labels = [f"{class_names[int(l)]} {s:.2f}" for l, s in zip(pred_labels_idx, pred_scores)]

    img_to_draw = draw_bounding_boxes(
        high_res_img, 
        boxes=pred_boxes, 
        labels=pred_labels, 
        colors="red", 
        width=2*scale_factor, 
        font_size=10*scale_factor, 
        font=font
    )

    return img_to_draw.cpu().numpy().transpose(1, 2, 0)
    
    