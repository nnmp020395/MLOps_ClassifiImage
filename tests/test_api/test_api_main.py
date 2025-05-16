"""
Tests unitaires de l'API FastAPI (DINOv2 Classifier) à l'aide du framework `unittest`.

Ce fichier teste les endpoints suivants :
- POST /predict : vérifie la prédiction d'image (dandelion vs grass).
- POST /check_duplicate : vérifie si une image est déjà connue ou nouvelle (gestion du réentraînement).

Les dépendances critiques comme le modèle, les embeddings, et les interactions avec MinIO/S3 sont mockées.
"""

import io
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from PIL import Image

import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
SRC_DIR = os.path.join(ROOT_DIR, "src")
AIRFLOW_UTILS_DIR = os.path.join(ROOT_DIR, "airflow/dags/utils")

sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, AIRFLOW_UTILS_DIR)

from api.main import app

client = TestClient(app)


def generate_dummy_image():
    """
    Génère une image RGB fictive en mémoire au format JPEG pour les tests.

    Returns:
        bytes: Image encodée en mémoire.
    """
    img = Image.new("RGB", (224, 224), color="blue")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.read()


class TestAPI(unittest.TestCase):
    """
    Classe de test unitaire pour les endpoints de l'API DINOv2 Classifier.
    """

    @patch("api.main.model")
    @patch("api.main.transform")
    def test_predict_success(self, mock_transform, mock_model):
        """
        Teste le endpoint POST /predict avec une image valide.
        Vérifie que la réponse contient bien une prédiction ('dandelion' ou 'grass').
        """
        dummy_tensor = MagicMock()
        mock_transform.return_value = dummy_tensor

        mock_output = MagicMock()
        mock_output.argmax.return_value.item.return_value = 1  # Index 1 -> "grass"
        mock_model.return_value = mock_output
        mock_model.__call__.return_value = mock_output

        files = {
            "file": ("test.jpg", generate_dummy_image(), "image/jpeg")
        }

        response = client.post("/predict", files=files)
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.json()["prediction"], ["dandelion", "grass"])

    @patch("api.main.get_embedding", return_value=[0.1] * 128)
    @patch("api.main.is_duplicate", return_value=False)
    @patch("api.main.s3_client.upload_fileobj")
    def test_check_duplicate_new(self, mock_upload, mock_is_duplicate, mock_get_embedding):
        """
        Teste le endpoint POST /check_duplicate pour une image inconnue (non dupliquée).
        Vérifie que l'image est détectée comme nouvelle et téléversée.
        """
        files = {
            "file": ("test.jpg", generate_dummy_image(), "image/jpeg")
        }
        data = {
            "label": "dandelion"
        }

        response = client.post("/check_duplicate", files=files, data=data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "new")

    @patch("api.main.get_embedding", return_value=[0.1] * 128)
    @patch("api.main.is_duplicate", return_value=True)
    def test_check_duplicate_known(self, mock_is_duplicate, mock_get_embedding):
        """
        Teste le endpoint POST /check_duplicate pour une image connue (dupliquée).
        Vérifie que l'image est reconnue comme déjà existante.
        """
        files = {
            "file": ("test.jpg", generate_dummy_image(), "image/jpeg")
        }
        data = {
            "label": "grass"
        }

        response = client.post("/check_duplicate", files=files, data=data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "known")


if __name__ == "__main__":
    unittest.main()
