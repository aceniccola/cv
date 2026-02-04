import os
from datasets import load_dataset
from ultralytics import YOLO
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

data_path = "datasets/imagenet_data" 

# 2. Load the Model
model = YOLO("yolo11n-cls.pt") 

# 3. Train
results = model.train(
    data=data_path, 
    epochs=100, 
    imgsz=224,
    device=0  # Use 'cpu' if no GPU is available
)

# 4. Validation
val_results = model.val() 

print(val_results)