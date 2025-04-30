"""
Module principal pour l'API FastAPI.

Ce fichier initialise et configure l'API FastAPI, définit les routes et gère
les requêtes.
"""

import io
import time

import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile, Request, Response
from fastapi.routing import APIRoute
from fastapi.responses import JSONResponse
from PIL import Image
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST


from src.model import load_model

app = FastAPI(title="DINOv2 Classifier API")

# ------------------ DEVICE ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ LOAD MODEL ------------------
# On charge le modèle depuis le bucket S3
model = load_model("s3://image-dandelion-grass/model/dinov2_classifier.pth")
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