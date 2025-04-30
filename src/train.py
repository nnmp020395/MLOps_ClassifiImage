"""
Script d'entraînement local du modèle DINOv2 pour la classification d'images.

Ce script :
1. Charge les images depuis un bucket S3.
2. Effectue un split entraînement/validation.
3. Entraîne un modèle basé sur un backbone DINOv2 gelé.
4. Évalue le modèle et le sauvegarde en local sous forme de fichier .pth.
"""

import io
import logging
import random
import os

import s3fs
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader, Dataset, Subset



# ------------------ LOGGING ------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Bienvenu dans le script d'entraînement!")

# ------------------ CONFIG MINIO ------------------
minio_endpoint = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
minio_key = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
minio_secret = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")

fs = s3fs.S3FileSystem(
    key=minio_key,
    secret=minio_secret,
    client_kwargs={"endpoint_url": minio_endpoint}
)

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


# ------------------ DATASET PERSO ------------------
class S3ImageFolder(Dataset):
    """
    Dataset personnalisé pour lire les images depuis un bucket S3.

    Args:
        root_paths (list): Liste des chemins S3 pour chaque classe.
        transform (callable, optional): Transformations à appliquer sur les images.
    """

    def __init__(self, root_paths, transform=None, fs=None):
        """Initialise le dataset en scannant les répertoires S3 et en mappant \
            les classes."""
        self.s3 = fs or s3fs.S3FileSystem()
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        for label_idx, s3_path in enumerate(sorted(root_paths)):
            class_name = s3_path.rstrip("/").split("/")[-1]
            self.class_to_idx[class_name] = label_idx

            files = self.s3.ls(s3_path)
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((f, label_idx))

    def __len__(self):
        """Retourne le nombre total d’échantillons."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Récupère une image et son label à l'index donné.

        Args:
            idx (int): Index de l’image à récupérer.

        Returns:
            Tuple[Tensor, int]: Image transformée et son label associé.
        """
        path, label = self.samples[idx]
        with self.s3.open(path, "rb") as f:
            image = Image.open(io.BytesIO(f.read())).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ------------------ CHARGEMENT & SPLIT ------------------
logging.info("Chargement des données depuis S3 et split 70/30.")
s3_paths = ["s3://image-dandelion-grass/raw/dandelion", "s3://image-dandelion-grass/raw/grass"]
full_dataset = S3ImageFolder(s3_paths, transform=transform, fs=fs)


def stratified_split(dataset, split_ratio=0.7):
    """
    Effectue une séparation équilibrée du dataset entre entraînement et validation.

    Args:
        dataset (Dataset): Dataset à splitter.
        split_ratio (float): Ratio pour l'entraînement (ex: 0.7 = 70% entraînement).

    Returns:
        Tuple[Subset, Subset]: Deux sous-ensembles (train, validation).
    """
    label_to_indices = {}
    for idx, (_, label) in enumerate(dataset.samples):
        label_to_indices.setdefault(label, []).append(idx)

    train_indices, val_indices = [], []
    for label, indices in label_to_indices.items():
        random.shuffle(indices)
        split = int(len(indices) * split_ratio)
        train_indices.extend(indices[:split])
        val_indices.extend(indices[split:])

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


train_data, val_data = stratified_split(full_dataset)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

logging.info(f"Classes : {full_dataset.class_to_idx}")
logging.info(f"Nb images train: {len(train_data)}, val: {len(val_data)}")

# ------------------ MODÈLE ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dino_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

# Gèle les poids du backbone pour ne pas les entraîner
for param in dino_backbone.parameters():
    param.requires_grad = False


class DinoClassifier(nn.Module):
    """
    Classificateur basé sur un backbone DINOv2 avec une tête de classification.

    Args:
        backbone (nn.Module): Modèle de features (ex: DINOv2).
        num_classes (int): Nombre de classes de sortie.
    """

    def __init__(self, backbone, num_classes):
        """Initialise le classificateur avec un backbone gelé et une couche linéaire."""
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.embed_dim, num_classes)

    def forward(self, x):
        """
        Propagation avant.

        Args:
            x (Tensor): Entrée image.

        Returns:
            Tensor: Logits du classificateur.
        """
        with torch.no_grad():
            x = self.backbone(x)
        return self.head(x)


model = DinoClassifier(dino_backbone, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.head.parameters(), lr=lr)

# ------------------ ENTRAÎNEMENT ------------------
logging.info("Démarrage de l'entraînement.")

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

    acc = 100 * correct / len(train_data)
    logging.info(f"Epoch {epoch + 1}: Loss={total_loss:.2f}, Accuracy={acc:.2f}%")

# ------------------ VALIDATION ------------------
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

# ------------------ SAUVEGARDE ------------------
torch.save(model.state_dict(), "dinov2_classifier.pth")
logging.info("Modèle sauvegardé sous 'dinov2_classifier.pth'.")
