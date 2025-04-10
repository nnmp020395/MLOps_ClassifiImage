import os
import io
import torch
import torch.nn as nn
import numpy as np
import logging
import boto3
import s3fs
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader  # same as PIL.Image.open
from torch import optim

# Logging config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset personnalisé
class S3ImageFolder(Dataset):
    def __init__(self, s3_root, transform=None):
        self.s3_root = s3_root.rstrip('/')
        self.fs = s3fs.S3FileSystem()
        self.transform = transform
        self.samples = []
        self.classes = []

        # Lister les classes (dossier = nom de classe)
        class_dirs = self.fs.ls(self.s3_root)
        self.classes = sorted([os.path.basename(p) for p in class_dirs])

        # Index des classes
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Collecter les chemins vers toutes les images
        for cls_name in self.classes:
            class_path = f"{self.s3_root}/{cls_name}"
            image_paths = self.fs.ls(class_path)
            for img_path in image_paths:
                if img_path.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((img_path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with self.fs.open(path, 'rb') as f:
            image = Image.open(f).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Utiliser le Dataset personnalisé
root_train = "s3://image-dadelion-grass/train"
root_val = "s3://image-dadelion-grass/val"

train_data = S3ImageFolder(root_train, transform=transform)
val_data = S3ImageFolder(root_val, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=4)

logging.info(f"Mapping des classes : {train_data.class_to_idx}")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load DINOv2 backbone
dino_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

# Freeze backbone
for param in dino_backbone.parameters():
    param.requires_grad = False

# Model
class DinoClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.embed_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)
        return self.head(x)

model = DinoClassifier(dino_backbone, num_classes=2).to(device)
criteron = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.head.parameters(), lr=0.003)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criteron(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()

    acc = 100 * correct / len(train_data)
    logging.info(f"Epoch {epoch}: Loss={total_loss:.2f}, Accuracy={acc:.2f}%")

# Validation
model.eval()
val_correct, val_total = 0, 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        val_correct += (preds == labels).sum().item()
        val_total += labels.size(0)

val_acc = 100 * val_correct / val_total
logging.info(f"Validation Accuracy: {val_acc:.2f}%")

# Save model
torch.save(model.state_dict(), "dinov2_classifier.pth")
logging.info("Modèle sauvegardé sous 'dinov2_classifier.pth'.")
