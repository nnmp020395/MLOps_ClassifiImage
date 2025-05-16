"""
API FastAPI pour la classification d'images (DINOv2 Classifier).

Cette API fournit :
- /predict : Prédiction de l'étiquette d'une image ("dandelion" ou "grass").
- /check_duplicate : Vérification de doublons dans le bucket MinIO.
- /metrics : Exposition des métriques Prometheus.
"""

import io
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from uuid import uuid4

import boto3
import torch
import torchvision.transforms as transforms
from botocore.config import Config
from fastapi import FastAPI, File, Request, Response, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src.model import load_model

app = FastAPI(title="DINOv2 Classifier API")

# --- PATH CONFIGS ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
AIRFLOW_UTILS_DIR = os.path.join(CURRENT_DIR, "../airflow/dags/utils")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, AIRFLOW_UTILS_DIR)

from duplicate_detector import get_embedding, is_duplicate

# --- LOGGING ---
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/api.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# --- MINIO / S3 CONFIGS ---
minio_endpoint = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
minio_access_key = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
minio_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
region_name = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
bucket_name = "image-dandelion-grass"

boto_config = Config(connect_timeout=5, read_timeout=30)
s3_client = boto3.client(
    "s3",
    endpoint_url=minio_endpoint,
    aws_access_key_id=minio_access_key,
    aws_secret_access_key=minio_secret_key,
    region_name=region_name,
    use_ssl=False,
    config=boto_config,
)

# --- MODEL & TRANSFORM ---
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_latest_model_for_date(date_obj):
    """
    Recherche le modèle le plus récent pour une date donnée sur MinIO.

    Args:
        date_obj (datetime): Date cible.

    Returns:
        Tuple[str | None, int]: Clé S3 du modèle et son index, ou (None, -1) si aucun trouvé.
    """
    prefix = f"model/{date_obj.strftime('%Y-%m-%d')}/"
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if "Contents" not in response:
            return None, -1
        candidates = [
            (int(obj["Key"].split("_")[-1].replace(".pth", "")), obj["Key"])
            for obj in response["Contents"]
            if obj["Key"].endswith(".pth") and "dinov2_classifier_" in obj["Key"]
        ]
        if not candidates:
            return None, -1
        candidates.sort()
        return candidates[-1][1], candidates[-1][0]
    except Exception as e:
        logger.error(f"Erreur recherche modèle : {e}")
        return None, -1


def initialize_model():
    """
    Recherche et charge le modèle le plus récent disponible sur les 7 derniers jours.

    Returns:
        torch.nn.Module: Modèle PyTorch chargé et prêt pour l'inférence.

    Raises:
        FileNotFoundError: Si aucun modèle n'a été trouvé.
    """
    for i in range(8):
        object_key, last_i = find_latest_model_for_date(datetime.now() - timedelta(days=i))
        if last_i >= 0:
            break
    if last_i < 0 or object_key is None:
        raise FileNotFoundError("Aucun modèle trouvé dans les 7 derniers jours.")
    
    logger.info(f"Chargement du modèle : {object_key}")
    model_buffer = io.BytesIO()
    s3_client.download_fileobj(bucket_name, object_key, model_buffer)
    model_buffer.seek(0)
    model = load_model(model_buffer)
    model.to(device)
    model.eval()
    return model


if os.getenv("FASTAPI_ENV") != "test":
    model = initialize_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225]),
])
class_names = ["dandelion", "grass"]

# --- PROMETHEUS METRICS ---
REQUEST_COUNT = Counter("request_count", "App Request Count", ["method", "endpoint"])
REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency", ["endpoint"])


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """
    Middleware pour collecter des métriques Prometheus sur chaque requête HTTP.
    """
    start_time = time.time()
    response = await call_next(request)
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(time.time() - start_time)
    return response


@app.get("/")
def home():
    """
    Endpoint racine pour vérifier si l'API est opérationnelle.

    Returns:
        dict: Message de bienvenue.
    """
    return {"message": "Bienvenue sur l'API de prédiction DINOv2."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint pour effectuer une prédiction d'image.

    Args:
        file (UploadFile): Image JPEG envoyée par l'utilisateur.

    Returns:
        dict: Étiquette prédite ("dandelion" ou "grass").
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(img_tensor).argmax(1).item()
            class_label = class_names[prediction]

        return {"prediction": class_label}

    except Exception as e:
        logger.error(f"Erreur prédiction : {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/check_duplicate")
async def check_duplicate(file: UploadFile = File(...), label: str = Form(...)):
    """
    Endpoint pour vérifier si l'image existe déjà dans la base (doublon).

    Si l'image est nouvelle, elle est uploadée dans MinIO sous `raw/new_data/pending_validation/`.

    Args:
        file (UploadFile): Image à vérifier.
        label (str): Label associé à l'image.

    Returns:
        dict: Statut ("known" ou "new") et message explicatif.
    """
    try:
        contents = await file.read()
        embedding = get_embedding(contents)

        if is_duplicate(embedding, s3_client, bucket_name):
            return {"status": "known", "message": "Image connue, pas de réentraînement."}
        else:
            buffer = io.BytesIO(contents)
            filename = f"{uuid4().hex}_{label}.jpg"
            s3_key = f"raw/new_data/pending_validation/{filename}"
            s3_client.upload_fileobj(buffer, bucket_name, s3_key)
            logger.info(f"Image téléversée : {s3_key}")
            return {"status": "new", "message": "Image ajoutée et téléversée dans MinIO."}

    except Exception as e:
        logger.error(f"Erreur doublon : {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/metrics")
def metrics():
    """
    Endpoint d'exposition des métriques Prometheus.

    Returns:
        Response: Contenu Prometheus lisible par un collecteur.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
