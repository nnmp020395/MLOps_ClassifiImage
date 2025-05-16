"""
Module de détection de doublons d'images basé sur des embeddings extraits via ResNet18.
Il permet d'obtenir des embeddings d'images, de vérifier si une image est un doublon
dans un bucket S3, et de maintenir un index local d'images pour un usage en mémoire.

Fonctionnalités :
- Extraction d'embeddings avec un modèle ResNet18 pré-entraîné.
- Détection de doublons en comparant les similarités cosinus.
- Indexation locale d'embeddings d'images.
"""

import logging
import os
from io import BytesIO
from uuid import uuid4

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ---------- LOGGING ----------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------- TRANSFORMATIONS POUR RESNET ----------
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# ---------- MODÈLE RESNET ----------
logger.info("Chargement du modèle ResNet18...")
resnet = models.resnet18(pretrained=True)
resnet = torch.nn.Sequential(
    *list(resnet.children())[:-1]
)  # Retire la dernière couche de classification
resnet.eval()

# ---------- INDEX LOCAL EN MÉMOIRE ----------
embedding_index = []
filename_index = []


# ---------- FONCTIONS ----------
def get_embedding(image_bytes: bytes) -> np.ndarray:
    """
    Extrait l'embedding d'une image à partir de ses bytes en utilisant ResNet18.

    Args:
        image_bytes (bytes): Image sous forme binaire.

    Returns:
        np.ndarray: Embedding de l'image sous forme de vecteur numpy.
    """
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = resnet(tensor).squeeze().numpy()
    return embedding.astype(np.float32)


def is_duplicate(
    embedding: np.ndarray, s3_client, bucket_name, prefix="raw/", threshold: float = 0.9
) -> bool:
    """
    Vérifie si un embedding correspond à un doublon dans un bucket S3 en comparant
    la similarité cosinus avec les images déjà présentes.

    Args:
        embedding (np.ndarray): Embedding de l'image à vérifier.
        s3_client: Client S3 boto3 configuré.
        bucket_name (str): Nom du bucket S3.
        prefix (str, optional): Préfixe du chemin dans le bucket. Par défaut "raw/".
        threshold (float, optional): Seuil de similarité pour considérer un doublon. Par défaut 0.9.

    Returns:
        bool: True si un doublon est détecté, sinon False.
    """
    embedding = embedding / np.linalg.norm(embedding)

    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if "Contents" not in response:
            return False

        for obj in response["Contents"]:
            key = obj["Key"]
            if not key.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            buffer = BytesIO()
            s3_client.download_fileobj(bucket_name, key, buffer)
            image_bytes = buffer.getvalue()
            other_embedding = get_embedding(image_bytes)
            other_embedding = other_embedding / np.linalg.norm(other_embedding)

            similarity = np.dot(embedding, other_embedding)
            if similarity >= threshold:
                logger.info(
                    f"Doublon détecté avec {key} (similarité = {similarity:.4f})"
                )
                return True

    except Exception as e:
        logger.error(f"Erreur dans la détection de doublon : {e}")

    return False


def add_image_to_index(filename: str, embedding: np.ndarray):
    """
    Ajoute une image et son embedding à l'index local en mémoire.

    Args:
        filename (str): Nom du fichier image.
        embedding (np.ndarray): Embedding associé à l'image.
    """
    embedding_index.append(embedding)
    filename_index.append(filename)
    logger.info(f"Image '{filename}' ajoutée à l'index local.")
