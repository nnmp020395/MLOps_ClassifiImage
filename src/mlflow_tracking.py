"""
Script pour l'entraînement et le suivi des expériences avec MLflow.

Ce fichier configure MLflow pour suivre les expériences d'entraînement d'un
modèle de classification d'images basé sur DinoV2. Les étapes incluent :
1. Configuration de l'URI de suivi MLflow.
2. Définition et mise à jour des expériences MLflow.
3. Enregistrement des métriques, paramètres et modèles.
"""

import io
import logging
import os
import random
import socket
import time

import mlflow
import mlflow.pytorch
import s3fs
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from mlflow.tracking import MlflowClient
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader, Dataset

# ------------------ LOGGING ------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Bienvenue dans le script d'entraînement!")

# ------------------ CONTEXTE ------------------
if "AIRFLOW_CTX_DAG_ID" in os.environ:
    run_name = "dag_run"
else:
    run_name = "local_run"

# ------------------ MLFLOW CONFIG ------------------
mlflow.set_tracking_uri("http://137.194.250.29:5001")
mlflow.set_experiment("DINOv2_Classifier")

# Ajout de la description de l'expérience
experiment_name = "DINOv2_Classifier"
description = "Classification d'images entre 'dandelion' et 'grass' avec DINOv2.\n"
client = MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)
if experiment:
    client.update_experiment(experiment.experiment_id, description=description)
    logging.info(f"Description de l'expérience '{experiment_name}' mise à jour.")
else:
    logging.warning(f"L'expérience '{experiment_name}' n'existe pas.")

# ------------------ HYPERPARAMÈTRES ------------------
batch_size = 32
lr = 0.003
num_epochs = 10

# ------------------ TRANSFORMATIONS ------------------
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def split_samples(s3_root, classes, split_ratio=0.7):
    """
    Divise les images S3 en deux listes : entraînement et validation.

    Args:
        s3_root (str): Chemin racine dans S3.
        classes (list): Liste des classes (ex: ['dandelion', 'grass']).
        split_ratio (float): Proportion des données pour l'entraînement.

    Returns:
        tuple: train_samples, val_samples, class_to_idx
    """
    fs = s3fs.S3FileSystem()
    train_samples, val_samples = [], []
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    for cls in classes:
        class_path = f"{s3_root}/{cls}"
        files = [
            f
            for f in fs.ls(class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        random.shuffle(files)
        split_idx = int(len(files) * split_ratio)
        train_samples += [(f, class_to_idx[cls]) for f in files[:split_idx]]
        val_samples += [(f, class_to_idx[cls]) for f in files[split_idx:]]

    return train_samples, val_samples, class_to_idx


class S3ImageFolder(Dataset):
    """
    Dataset personnalisé pour charger des images depuis S3.

    Attributes:
        samples (list): Liste de tuples (chemin S3, étiquette).
        transform (callable): Transformations à appliquer aux images.
    """

    def __init__(self, samples, transform=None):
        """
        Initialise le dataset S3ImageFolder.

        Args:
            samples (list): Liste de tuples (path, label).
            transform (callable, optional): Transformations PyTorch.
        """
        self.s3 = s3fs.S3FileSystem()
        self.samples = samples
        self.transform = transform

    def __len__(self):
        """Retourne le nombre d'échantillons."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Charge une image à partir de S3 et applique les transformations.

        Args:
            idx (int): Index de l'échantillon.

        Returns:
            tuple: image transformée, label
        """
        path, label = self.samples[idx]
        with self.s3.open(path, "rb") as f:
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
        return self.transform(img), label if self.transform else img, label


# ------------------ CHARGEMENT DES DONNÉES ------------------
s3_root = "s3://image-dadelion-grass"
classes = ["dandelion", "grass"]
train_samples, val_samples, class_to_idx = split_samples(s3_root, classes)

logging.info(f"Classes trouvées : {class_to_idx}")
logging.info(f"Nb images train: {len(train_samples)}, val: {len(val_samples)}")

train_dataset = S3ImageFolder(train_samples, transform)
val_dataset = S3ImageFolder(val_samples, transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ------------------ MODÈLE ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dino_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

for param in dino_backbone.parameters():
    param.requires_grad = False


class DinoClassifier(nn.Module):
    """
    Classificateur DINOv2 personnalisé.

    Args:
        backbone (nn.Module): Backbone pré-entraîné (ex: DINOv2).
        num_classes (int): Nombre de classes de sortie.
    """

    def __init__(self, backbone, num_classes):
        """Initialise le classificateur avec un backbone et une couche de sortie."""
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.embed_dim, num_classes)

    def forward(self, x):
        """
        Applique le backbone DINO suivi de la couche de classification.

        Args:
            x (torch.Tensor): Batch d’images.

        Returns:
            torch.Tensor: Logits de classification.
        """
        with torch.no_grad():
            x = self.backbone(x)
        return self.head(x)


model = DinoClassifier(dino_backbone, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.head.parameters(), lr=lr)

# ------------------ ENTRAÎNEMENT ------------------
start_time = time.time()

with mlflow.start_run(run_name=run_name):
    mlflow.set_tags({"source": run_name, "host": socket.gethostname()})

    mlflow.log_params(
        {
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": num_epochs,
            "backbone": "dinov2_vits14",
        }
    )

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()

        acc = 100 * correct / len(train_dataset)
        mlflow.log_metric("train_accuracy", acc, step=epoch)
        mlflow.log_metric("train_loss", total_loss, step=epoch)
        logging.info(
            f"Epoch {epoch:02d} - Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%"
        )

    # ------------------ VALIDATION ------------------
    logging.info("Évaluation du modèle...")
    model.eval()
    val_correct, val_total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total
    mlflow.log_metric("val_accuracy", val_acc)
    logging.info(f"Validation Accuracy = {val_acc:.2f}%")

    # ------------------ SAVE MODEL ------------------
    mlflow.pytorch.log_model(model, artifact_path="model")

    fs = s3fs.S3FileSystem()
    with fs.open("image-dadelion-grass/model/dinov2_classifier.pth", "wb") as f:
        torch.save(model.state_dict(), f)

    logging.info(
        "Modèle loggé avec MLflow et sauvegardé dans le bucket S3 sous \
            'model/dinov2_classifier.pth'."
    )

    # ------------------ DURATION ------------------
    duration = time.time() - start_time
    mlflow.log_metric("duration", duration)
    logging.info(f"Durée d'exécution : {duration:.2f} secondes.")

mlflow.end_run()
