"""
Module pour gérer l'entrainement et le ré-entrainement du modèle DINOv2.

Ce module contient les fonctions nécessaires pour :
1. Vérifier la présence de nouvelles images images dans le bucket MinIO.
2. Lancer le script de suivi d'expérience avec MLflow.
"""

import logging
import os
import subprocess

import boto3
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

# ------------------ CONFIGURATION ------------------
BUCKET = "image-dandelion-grass"
PREFIX_RAW = "raw/new_data/validated_data/"
MINIO_CLIENT = boto3.client(
    "s3",
    endpoint_url=os.getenv("MINIO_ENDPOINT", "http://minio:9000"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"),
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    use_ssl=False,
)


def check_minio_image_count() -> bool:
    """
    Vérifie si au moins 10 images sont présentes dans le bucket MinIO.
    Retourne True si c'est le cas, sinon False.
    """
    response = MINIO_CLIENT.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX_RAW)
    images = [
        obj["Key"]
        for obj in response.get("Contents", [])
        if obj["Key"].lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    count = len(images)
    logging.info(f"Nombre d'images trouvées : {count} dans {PREFIX_RAW}")

    if count >= 10:
        logging.info(
            "Au moins 10 nouvelles images trouvées. Lancement du script de suivi MLFlow."
        )
        return True
    else:
        logging.info("Pas assez d'images pour déclencher l'entraînement.")
        return False


def run_mlflow_tracking():
    """
    Lance le script de tracking MLFlow.
    """
    logging.info("Lancement du script mlflow_tracking.py via subprocess.")
    try:
        result = subprocess.run(
            ["python", "/opt/airflow/src/mlflow_tracking.py"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        logging.info("Script exécuté avec succès.")
        logging.info("STDOUT:\n" + result.stdout)
        logging.info("STDERR:\n" + result.stderr)
    except subprocess.CalledProcessError as e:
        logging.error("Échec du script mlflow_tracking.py.")
        logging.error("STDOUT:\n" + e.stdout)
        logging.error("STDERR:\n" + e.stderr)
        raise e
