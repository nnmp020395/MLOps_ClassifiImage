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
from datetime import datetime

import boto3
# import matplotlib.pyplot as plt
# import mlflow.pytorch
import s3fs
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from mlflow.tracking import MlflowClient
from PIL import Image
# from sklearn.metrics import auc, roc_curve
from torch import optim
from torch.utils.data import DataLoader, Dataset

import mlflow

os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MINIO_ENDPOINT", "http://minio:9000")


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
mlflow.set_tracking_uri("http://mlflow-server:6000")
mlflow.set_experiment("DINOv2_Classifier")

# Ajout de la description de l'expérience
experiment_name = "DINOv2_Classifier"
description = "Classification d'images entre 'dandelion' et 'grass' avec DINOv2.\n"
client = MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)
if experiment:
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
    fs = s3fs.S3FileSystem(
        key=os.getenv("AWS_ACCESS_KEY_ID"),
        secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
        client_kwargs={
            "endpoint_url": os.getenv("MINIO_ENDPOINT", "http://minio:9000")
        },
    )
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
        self.s3 = s3fs.S3FileSystem(
            key=os.getenv("AWS_ACCESS_KEY_ID"),
            secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
            client_kwargs={
                "endpoint_url": os.getenv("MINIO_ENDPOINT", "http://minio:9000")
            },
        )
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
        if self.transform:
            img = self.transform(img)
        return img, label


# ------------------ CHARGEMENT DES DONNÉES ------------------
s3_root = "s3://image-dandelion-grass/raw"
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
    all_labels, all_outputs = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            all_labels.extend(labels.cpu().numpy())
            outputs = model(images)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total
    mlflow.log_metric("val_accuracy", val_acc)
    logging.info(f"Validation Accuracy = {val_acc:.2f}%")

    # ------------------ ROC CURVE ------------------
    # fpr, tpr, _ = roc_curve(all_labels, all_outputs)
    # roc_auc = auc(fpr, tpr)
    # mlflow.log_metric("roc_auc", roc_auc)
    # logging.info(f"AUC: {roc_auc:.4f}")

    # # Plot the ROC curve
    # plt.figure()
    # plt.plot(fpr, tpr, color="blue", label=f"AUC = {roc_auc:.4f}")
    # plt.plot([0, 1], [0, 1], color="k", linestyle="--")
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("ROC Curve")
    # plt.legend(loc="lower right")
    # plt.savefig("roc_curve.png")
    # plt.close()

    # mlflow.log_artifact("roc_curve.png", artifact_path="roc_curve")
    # logging.info("Courbe ROC as an ")

    # ------------------ SAVE MODEL ------------------
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"),
        endpoint_url=os.getenv("MINIO_ENDPOINT", "http://minio:9000"),
    )

    bucket_name = "image-dandelion-grass"

    # Date du jour
    date_str = datetime.now().strftime("%Y-%m-%d")
    prefix = f"model/{date_str}/"
    i = 0

    # Liste des objets existants dans le dossier du jour
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    existing_keys = [obj["Key"] for obj in response.get("Contents", [])]

    # Trouve le prochain nom de fichier disponible
    while f"{prefix}dinov2_classifier_{i}.pth" in existing_keys:
        i += 1

    object_key = f"{prefix}dinov2_classifier_{i}.pth"

    # Sauvegarde locale temporaire
    local_model_path = "/tmp/dinov2_classifier.pth"
    torch.save(model.state_dict(), local_model_path)

    # Upload vers S3/MinIO
    s3_client.upload_file(local_model_path, bucket_name, object_key)

    os.remove(local_model_path)

    logging.info(f"Modèle sauvegardé sur MinIO à s3://{bucket_name}/{object_key}")

    # ------------------ DURATION ------------------
    duration = time.time() - start_time
    mlflow.log_metric("duration", duration)
    logging.info(f"Durée d'exécution : {duration:.2f} secondes.")

mlflow.end_run()
