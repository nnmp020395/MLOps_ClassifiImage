import os
import torch
import torch.nn as nn
import torchvision
import numpy as np
import logging
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import optim
import mlflow
import mlflow.pytorch

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configuration MLflow
mlflow.set_tracking_uri("http://localhost:5000")  
mlflow.set_experiment("DINOv2_Classifier")

# Hyperparamètres à logger dans MLflow
batch_size = 32
lr = 0.003
num_epochs = 10

#######################################
# Train script
#######################################

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
])

# Données
root_train = '/path/to/train/'
root_val = '/path/to/val/'
train_data = ImageFolder(root=root_train, transform=transform)
val_data = ImageFolder(root=root_val, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dino_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

# Geler le backbone
for param in dino_backbone.parameters():
    param.requires_grad = False

# Modèle personnalisé
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

model = DinoClassifier(dino_backbone, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.head.parameters(), lr=lr)

#######################################
# Lancer le run MLflow
########################################

with mlflow.start_run():
    # Log des hyperparamètres
    mlflow.log_params({
        "learning_rate": lr,
        "batch_size": batch_size,
        "epochs": num_epochs,
        "backbone": "dinov2_vits14"
    })

    # Entraînement
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()

        train_acc = 100 * correct / len(train_data)
        mlflow.log_metric("train_accuracy", train_acc, step=epoch)
        mlflow.log_metric("train_loss", total_loss, step=epoch)
        logging.info(f"Epoch {epoch} - Loss: {total_loss:.2f}, Accuracy: {train_acc:.2f}%")

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total
    mlflow.log_metric("val_accuracy", val_acc)
    logging.info(f"Validation Accuracy = {val_acc:.2f}%")

    # Enregistrement du modèle
    mlflow.pytorch.log_model(model, artifact_path="model")
    logging.info("Modèle loggé avec MLflow.")
