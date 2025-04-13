"""
Module pour effectuer des prédictions avec le modèle DinoV2.

Ce fichier charge un modèle pré-entraîné et fournit une fonction `predict`
pour effectuer des prédictions sur des images. Les étapes incluent :
1. Chargement du modèle entraîné.
2. Prétraitement des images.
3. Prédiction de la classe de l'image.
"""
import torch
import torchvision.transforms as transforms
from PIL import Image

from model import load_model

# ------------------ MODEL LOADING ------------------
# Chargement du modèle entraîné
model = load_model("dinov2_classifier.pth")

# ------------------ TRANSFORM ------------------
# Définition de la transformation de prétraitement à appliquer sur l'image
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Détection de l'appareil (GPU ou CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(image_path):
    """
    Effectue une prédiction de classe sur une image fournie.

    Paramètres
    ----------
    image_path : str
        Chemin vers l'image à classer.

    Retourne
    -------
    str
        Le label prédit : "dandelion" ou "grass".
    """
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)  # Ajoute la dimension batch
    output = model(image)
    predicted_class = output.argmax(dim=1).item()

    class_names = ["dandelion", "grass"]
    print(f"Predicted Class: {class_names[predicted_class]}")

    return class_names[predicted_class]
