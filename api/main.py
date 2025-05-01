"""
Module principal pour l'API FastAPI.

Ce fichier initialise et configure l'API FastAPI, définit les routes et gère
les requêtes.
"""

import io
import os
import time
from datetime import datetime, timedelta

import boto3
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Histogram,
                               generate_latest)

from src.model import load_model

app = FastAPI(title="DINOv2 Classifier API")

# ------------------ DEVICE ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ CONFIG MINIO ------------------
minio_endpoint = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
minio_access_key = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
minio_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
region_name = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
bucket_name = "image-dandelion-grass"

# ------------------ CLIENT S3 (MinIO) ------------------
s3_client = boto3.client(
    "s3",
    endpoint_url=minio_endpoint,
    aws_access_key_id=minio_access_key,
    aws_secret_access_key=minio_secret_key,
    region_name=region_name,
    use_ssl=False,
)


# ------------------ FUNCTION ------------------
def find_latest_model_for_date(date_obj):
    """
    Recherche le modèle le plus récent pour une date donnée dans le bucket S3.

    Args:
        date_obj (datetime): Date pour laquelle rechercher le modèle.
    Returns:
        tuple: Clé de l'objet S3 et indice du modèle le plus récent.
    """
    prefix = f"model/{date_obj.strftime('%Y-%m-%d')}/"
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if "Contents" not in response:
            return None, 0

        print(
            f"Fichiers trouvés ({date_obj}):",
            [obj["Key"] for obj in response["Contents"]],
        )

        indices = []
        for obj in response["Contents"]:
            key = obj["Key"]
            filename = key.split("/")[-1]
            print(f"Nom de fichier: {filename}")

            if filename.startswith("dinov2_classifier_") and filename.endswith(".pth"):
                index = int(filename[len("dinov2_classifier_") : -len(".pth")])
                indices.append((index, key))

        if not indices:
            return None, 0

        indices.sort()
        best_key = indices[-1][1]
        return best_key, indices[-1][0]

    except Exception as e:
        print(f"Erreur lors de la recherche du modèle : {e}")
        return None, 0


# ------------------ CHARGEMENT DU MODÈLE ------------------

# Essayer aujourd'hui
today = datetime.now()
object_key, last_i = find_latest_model_for_date(today)

# Sinon essayer hier
if last_i == 0:
    yesterday = today - timedelta(days=1)
    object_key, last_i = find_latest_model_for_date(yesterday)

    if last_i == 0 and object_key is None:
        raise FileNotFoundError(
            "Aucun modèle trouvé dans les dossiers des deux derniers jours!"
        )

# Téléchargement en mémoire
print(f"Chargement du modèle : {object_key}")
model_buffer = io.BytesIO()
s3_client.download_fileobj(bucket_name, object_key, model_buffer)
model_buffer.seek(0)

# Chargement dans PyTorch
model = load_model(model_buffer)
model.to(device)
model.eval()

# ------------------ TRANSFORM ------------------
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# ------------------ CLASS LABELS ------------------
class_names = ["dandelion", "grass"]


# ------------------ ROUTES ------------------
REQUEST_COUNT = Counter("request_count", "App Request Count", ["method", "endpoint"])
REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency", ["endpoint"])


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    endpoint = request.url.path
    method = request.method

    REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(process_time)

    return response


@app.get("/")
def home():
    """
    Route d'accueil de l'API.

    Returns:
        dict: Un message de bienvenue.
    """
    return {"message": "Bienvenue sur l'API de prédiction DINOv2."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Effectue une prédiction de classe (dandelion ou grass) à partir d'une image envoyée.

    Args:
        file (UploadFile): Fichier image téléversé par l'utilisateur (jpg, jpeg ou png).

    Returns:
        dict: Résultat de la prédiction sous forme de classe ou message d'erreur.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            predicted = outputs.argmax(1).item()
            class_label = class_names[predicted]

        return {"prediction": class_label}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
