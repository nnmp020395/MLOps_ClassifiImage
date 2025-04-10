import os
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import logging
import s3fs
from torch import optim
import mlflow
import mlflow.pytorch

# ------------------ LOGGING ------------------
logging.basicConfig(level=logging.INFO, \
                    format="%(asctime)s - %(levelname)s - %(message)s")

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

# ------------------ CUSTOM S3 DATASET ------------------
class S3ImageFolder(Dataset):
    def __init__(self, s3_root, transform=None):
        self.s3 = s3fs.S3FileSystem()
        self.root = s3_root.rstrip('/')
        self.transform = transform
        self.classes = sorted(self.s3.ls(self.root))
        self.class_to_idx = {cls.split('/')[-1]: i for i, \
                             cls in enumerate(self.classes)}
        self.samples = []

        for class_path in self.classes:
            files = self.s3.ls(class_path)
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((f, \
                        self.class_to_idx[class_path.split("/")[-1]]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with self.s3.open(path, 'rb') as f:
            img = Image.open(io.BytesIO(f.read())).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# ------------------ LOAD DATA FROM S3 ------------------
root_train = "s3://image-dadelion-grass/train"
root_val = "s3://image-dadelion-grass/val"

train_data = S3ImageFolder(s3_root=root_train, transform=transform)
val_data = S3ImageFolder(s3_root=root_val, transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

logging.info(f"Classes trouvées : {train_data.class_to_idx}")

# ------------------ DEVICE & MODEL ------------------
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
        x = self.head(x)
        return x

model = DinoClassifier(dino_backbone, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.head.parameters(), lr=lr)

# ------------------ TRAINING LOOP ------------------
with mlflow.start_run():

    # Enregistrement des hyperparamètres
    mlflow.log_params({
        "learning_rate": lr,
        "batch_size": batch_size,
        "epochs": num_epochs,
        "backbone": "dinov2_vits14"
    })

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

        # Enregistrement des métriques
        mlflow.log_metric("train_accuracy", train_acc, step=epoch)
        mlflow.log_metric("train_loss", total_loss, step=epoch)
        
        logging.info(f"Epoch {epoch} - Loss: {total_loss:.2f}, \
                     Accuracy: {train_acc:.2f}%")

    # ------------------ VALIDATION ------------------
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

    # ------------------ SAVE MODEL ------------------
    mlflow.pytorch.log_model(model, artifact_path="model")
    logging.info("Modèle loggé avec MLflow.")
