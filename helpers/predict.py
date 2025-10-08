import torch
from pathlib import Path
from PIL import Image
from models.YOLOBase import YOLOBase
import torchvision.transforms as T

# Paths
runs_folder = Path('runs/run_1')
dataset_folder = Path('dataset128/test/images')
model_path = runs_folder / 'best.pt'  # Adjust if your model filename is different
image_path = dataset_folder / '4ebb81ab-4f6a-46db-86e3-42b2acfe78ce_png.rf.e18b9a73aa6be28f7ed6663b9e54e9be.jpg'  # Adjust to your image filename

# Load model
model = YOLOBase(nc=46)
state_dict = torch.load(model_path, map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

# Load image
img = Image.open(image_path)
img = img.convert('RGB')

# Detect
results = model(img)

# Show results
results.show()