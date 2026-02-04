import os
from datasets import load_dataset
from ultralytics import YOLO
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# 1. Load and Prepare the Dataset
# Note: ImageNet requires a login/access on HF. 
# For a quick test, you can use "zh-plus/tiny-imagenet"
dataset_name = "ILSVRC/imagenet-1k"
ds = load_dataset(dataset_name, storage_options={'anon': True}) 

# YOLO needs physical files. You must save the HF dataset to a local folder
# formatted as: data/train, data/val, data/test
def save_to_yolo_format(dataset, root_path):
    for split in dataset.keys():
        split_path = os.path.join(root_path, split)
        dataset[split].save_to_disk(split_path)
    return root_path

save_to_yolo_format(ds, "")