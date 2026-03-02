import os
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# ==========================
# CONFIG
# ==========================

DATA_DIR = Path("data/plantvillage_small")
IMG_SIZE = 224
VAL_SPLIT = 0.2
BATCH_SIZE = 32


# =====================
# DEVICE
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================
# TRANSFORMS
# =====================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])

# =====================
# DATASET
# =====================
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

num_classes = len(dataset.classes)
print("Classes:", dataset.classes)

# Train/Val split
val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =====================
# MODEL
# =====================
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model = model.to(device)