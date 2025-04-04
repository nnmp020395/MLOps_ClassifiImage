import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import numpy as np
import logging
from collections import OrderedDict
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import optim

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Transformations pour prétraitement des images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
])

# Chargement du dataset depuis le dossier (à adapter pour S3 si besoin)
root_train = '/Users/phuongnguyen/Documents/cours_BGD_Telecom_Paris_2024/712_MLOps/dataset_project/train/'
root_val = '/Users/phuongnguyen/Documents/cours_BGD_Telecom_Paris_2024/712_MLOps/dataset_project/val/'
train_data = ImageFolder(root=root_train, transform=transform)
val_data = ImageFolder(root=root_val, transform=transform)

# Création des DataLoaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Affichage du mapping classe -> index
logging.info(f"Mapping des classes : {train_data.class_to_idx}")

# Choix du device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chargement du backbone DINOv2
dino_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

# Geler tous les paramètres du backbone
for param in dino_backbone.parameters():
    param.requires_grad = False

# Définition du modèle de classification
class DinoClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.embed_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)
        x = self.head(x)
        return x

# Fonction de perte et optimiseur
criteron = nn.CrossEntropyLoss()
model = DinoClassifier(dino_backbone, num_classes=2).to(device)
lr = 0.003
optimizer = optim.Adam(params=model.head.parameters(), lr=lr)

# Entraînement
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criteron(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()

    train_acc = 100 * correct / len(train_data)
    logging.info(f"Learning rate : {lr} - Époque {epoch} : Perte = {total_loss:.2f}, Précision = {train_acc:.2f}%")

# Validation
model.eval()
val_correct = 0
val_total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = outputs.argmax(dim=1)

        val_correct += (predicted == labels).sum().item()
        val_total += labels.size(0)

val_acc = 100 * val_correct / val_total
logging.info(f"Précision sur la validation : {val_acc:.2f}%")

# Sauvegarde du modèle
torch.save(model.state_dict(), 'dinov2_classifier.pth')
logging.info("Modèle sauvegardé sous 'dinov2_classifier.pth'.")

# Upload vers S3 (optionnel)
# import boto3
# s3 = boto3.client('s3')
# s3.upload_file('dinov2_classifier.pth', 'dinov2-model', 'dinov2_classifier.pth')
# logging.info("Modèle uploadé sur S3.")
