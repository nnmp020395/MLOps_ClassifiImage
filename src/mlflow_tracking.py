import os
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import logging
import s3fs
import random
from torch import optim
import mlflow
import mlflow.pytorch
import socket
import time


# ------------------ LOGGING ------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Bienvenu dans le script d'entraînement!")

# Détection du contexte d'exécution
if "AIRFLOW_CTX_DAG_ID" in os.environ:
    run_name = "dag_run"
else:
    run_name = "local_run"

# ------------------ MLFLOW CONFIG ------------------
mlflow.set_tracking_uri("http://137.194.250.29:5001")
mlflow.set_experiment("DINOv2_Classifier")

# ------------------ HYPERPARAMETERS ------------------
batch_size = 32
lr = 0.003
num_epochs = 10

# ------------------ TRANSFORMATIONS ------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
])

# ------------------ SPLIT FUNCTION ------------------
def split_samples(s3_root, classes, split_ratio=0.7):
    fs = s3fs.S3FileSystem()
    train_samples = []
    val_samples = []
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    for cls in classes:
        class_path = f"{s3_root}/{cls}"
        files = [f for f in fs.ls(class_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        random.shuffle(files)
        split_idx = int(len(files) * split_ratio)
        train_samples += [(f, class_to_idx[cls]) for f in files[:split_idx]]
        val_samples += [(f, class_to_idx[cls]) for f in files[split_idx:]]

    return train_samples, val_samples, class_to_idx

# ------------------ DATASET CLASS ------------------
class S3ImageFolder(Dataset):
    def __init__(self, samples, transform=None):
        self.s3 = s3fs.S3FileSystem()
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with self.s3.open(path, 'rb') as f:
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# ------------------ LOAD DATA ------------------
s3_root = "s3://image-dadelion-grass"
classes = ["dandelion", "grass"]
train_samples, val_samples, class_to_idx = split_samples(s3_root, classes)

logging.info(f"Classes trouvées : {class_to_idx}")
logging.info(f"Nb images train: {len(train_samples)}, val: {len(val_samples)}")

train_dataset = S3ImageFolder(train_samples, transform)
val_dataset = S3ImageFolder(val_samples, transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ------------------ MODEL ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dino_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

for param in dino_backbone.parameters():
    param.requires_grad = False

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
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.head.parameters(), lr=lr)

# ------------------ TRAINING ------------------
start_time = time.time()

with mlflow.start_run(run_name=run_name):

    mlflow.set_tags({
        "source": run_name,
        "host": socket.gethostname()
    })

    mlflow.log_params({
        "learning_rate": lr,
        "batch_size": batch_size,
        "epochs": num_epochs,
        "backbone": "dinov2_vits14"
    })

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
        logging.info(f"Epoch {epoch:02d} - Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")

    # ------------------ VALIDATION ------------------
    logging.info("\n Évaluation du modèle...")
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
    logging.info("Modèle loggé avec MLflow.")

    # ------------------ DURATION ------------------
    duration = time.time() - start_time
    mlflow.log_metric("duration", duration)
    logging.info(f"Durée d'exécution du run : {duration:.2f} secondes.")

mlflow.end_run()
