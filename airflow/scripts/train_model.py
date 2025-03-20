import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def train_pytorch_model(data_dir="../clean_dataset", model_path="../model.pth", batch_size=32, epochs=5, learning_rate=0.001):
    """
    Entraîne un modèle de classification d'images en utilisant ResNet18 avec logging et debug.

    Args:
        data_dir (str): Chemin vers le dossier contenant les données.
        model_path (str): Chemin de sauvegarde du modèle entraîné.
        batch_size (int): Taille du batch pour l'entraînement.
        epochs (int): Nombre d'époques d'entraînement.
        learning_rate (float): Taux d'apprentissage.

    Returns:
        None
    """
    try:
        # Vérification du GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Utilisation du device : {device}")

        # Vérification du dataset
        if not os.path.exists(data_dir):
            logging.error(f"Le dossier du dataset '{data_dir}' n'existe pas")
            return

        # Transformation et pré-traitement des images
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        # Chargement des données avec gestion des erreurs
        try:
            train_data = datasets.ImageFolder(root=data_dir, transform=transform)
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4 if torch.cuda.is_available() else False)
            logging.info(f"Dataset chargé avec {len(train_data)} images.")
        except Exception as e:
            logging.error(f"Erreur lors du chargement des données : {e}")
            return

        # Vérification du nombre de classes
        num_classes = len(train_data.classes)
        logging.info(f"Nombre de classes détectées : {num_classes} ({train_data.classes})")
        if num_classes != 2:
            logging.warning("Le modèle est conçu pour 2 classes, mais d'autres classes ont été détectées.")

        # Chargement du modèle ResNet18 pré-entraîné
        try:
            model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)  # Adaptation aux classes du dataset
            model = model.to(device)
            logging.info("Modèle ResNet18 chargé avec succès.")
        except Exception as e:
            logging.error(f"Erreur lors du chargement du modèle : {e}")
            return

        # Définition de la fonction de perte et de l'optimiseur
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Entraînement du modèle
        logging.info("Début de l'entraînement du modèle...")
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                logging.debug(f"Batch {batch_idx + 1}/{len(train_loader)} - Taille batch: {images.shape}")

                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                
                # Calcul de l'exactitude
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            accuracy = 100 * correct / total
            logging.info(f"Époque {epoch+1}/{epochs} - Perte: {running_loss/len(train_loader):.4f} - Précision: {accuracy:.2f}%")

        # Sauvegarde du modèle
        torch.save(model.state_dict(), model_path)
        logging.info(f"Modèle sauvegardé sous '{model_path}'.")

    except Exception as e:
        logging.error(f"Erreur générale dans l'entraînement : {e}")