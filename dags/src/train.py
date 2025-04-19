import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch import optim
from airflow.hooks.base import BaseHook
from sqlalchemy import text, create_engine
from airflow.hooks.base import BaseHook
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import requests
import boto3
import s3fs
import logging

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

# Load pre-trained DINOv2 model
logging.info("Loading DINOv2 model...")
dino_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
# Freeze all parameters except classification head
for param in dino_backbone.parameters():
    param.requires_grad = False
model = DinoClassifier(dino_backbone, num_classes=2).to(device)

def train(model, data_loader, num_epochs=10):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.head.parameters(), lr=0.003)
    for epoch in range(num_epochs):
        total_loss, correct = 0, 0
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
        train_acc = 100 * correct / len(data_loader)
        logging.info(f"Epoch {epoch}: Loss = {total_loss:.4f}, Accuracy = {train_acc:.2f}%")

def validate(model, data_loader):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)
    val_acc = 100 * total_correct / total_samples
    logging.info(f"Validation Accuracy = {val_acc:.2f}%")

def saving_model(model, path):
    return torch.save(model.state_dict(), path)

def main(mode='local'):

    logging.info("Load AWS credentials and initialize S3 client ...")
    # Récupérer les informations de connexion à S3 de la connexion Airflow
    conn = BaseHook.get_connection("aws_default")
    aws_access_id = conn.login
    aws_access_secret = conn.password

    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_id,
        aws_secret_access_key=aws_access_secret
    )
    logging.info('Connexion à AWS S3 ...')

    # Create S3 resource
    s3_resource = boto3.resource(
        "s3",
        aws_access_key_id=aws_access_id,
        aws_secret_access_key=aws_access_secret
    )

    # Initialisation
    S3_BUCKET_NAME = "image-dadelion-grass"
    S3_OUTPUT_MODEL =  f"s3a://{S3_BUCKET_NAME}/"

    if mode == 'local':
        # Local dataset
        root_train = '/Users/phuongnguyen/Documents/cours_BGD_Telecom_Paris_2024/712_MLOps/dataset_project/train/'
        root_val = '/Users/phuongnguyen/Documents/cours_BGD_Telecom_Paris_2024/712_MLOps/dataset_project/val/'
        # Load dataset from local directory
        train_folder = ImageFolder(root=root_train, transform=transform)
        val_folder = ImageFolder(root=root_val, transform=transform)
    elif mode == 's3':
        # S3 dataset
        s3_bucket_train = f"{S3_BUCKET_NAME}/train/"
        s3_bucket_val = f"{S3_BUCKET_NAME}/val/"
        # Load dataset directly from S3
        train_folder = ImageFolder(root=s3_bucket_train, transform=transform)
        val_folder = ImageFolder(root=s3_bucket_val, transform=transform)

    # Check class mappings
    logging.info(train_folder.class_to_idx)
    # Data loaders
    train_loader = DataLoader(train_folder, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_folder, batch_size=32, shuffle=False)

    # Train the model
    train(model, train_loader, num_epochs=10)
    # Validate the model
    validate(model, val_loader)
    logging.info("Training and validation completed.")

    # Save model locally
    saving_model(model, "dinov2_classifier.pth")
    logging.info("Saving model locally ...")

    # Upload model to S3
    s3.upload_file("./dinov2_classifier.pth", S3_OUTPUT_MODEL, "dinov2_classifier.pth")
    logging.info(f"Model uploaded to {S3_OUTPUT_MODEL}")

if __name__ == "__main__":
    mode = 'local'  # Change to 's3' if you want to load data from S3
    main(mode)
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
