import logging
import os
from io import BytesIO
from uuid import uuid4

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ---------- CONFIGURATION ----------
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
)  # Retirer la couche de classification
resnet.eval()

# ---------- INDEX LOCAL EN MÉMOIRE ----------
embedding_index = []
filename_index = []


# ---------- FONCTIONS UTILITAIRES ----------
def get_embedding(image_bytes: bytes) -> np.ndarray:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = resnet(tensor).squeeze().numpy()
    return embedding.astype(np.float32)


def is_duplicate(
    embedding: np.ndarray, s3_client, bucket_name, prefix="raw/", threshold: float = 0.9
) -> bool:
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
    embedding_index.append(embedding)
    filename_index.append(filename)
    logger.info(f"Image '{filename}' ajoutée à l'index local.")
