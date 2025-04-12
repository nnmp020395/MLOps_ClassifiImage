from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io
import torchvision.transforms as transforms
import sys
import os

from src.model import load_model


app = FastAPI(title="DINOv2 Classifier API")

# ------------------ DEVICE ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ LOAD MODEL ------------------
model = load_model("dinov2_classifier.pth") 
model.to(device)
model.eval()

# ------------------ TRANSFORM ------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
])

# ------------------ CLASS LABELS ------------------
class_names = ["dandelion", "grass"]

# ------------------ ROUTES ------------------
@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API de prédiction DINOv2."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
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
