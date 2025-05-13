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
from io import BytesIO

# ------------------ CONFIGURATION ------------------
BUCKET = "image-dandelion-grass"
PREFIX_RAW = "raw/new_data/corrected_data/"
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


def update_database():
    """
    Déplace les images validées depuis 'raw/new_data/corrected_data/' vers
    'raw/dandelion/' ou 'raw/grass/', selon leur nom.
    Le suffixe de classe (_dandelion ou _grass) est retiré du nom de fichier.
    """
    logging.info("Mise à jour de la base de données MinIO avec les images validées.")
    
    try:
        response = MINIO_CLIENT.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX_RAW)
        objects = response.get("Contents", [])
        image_keys = [obj["Key"] for obj in objects if obj["Key"].endswith((".jpg", ".jpeg", ".png"))]

        for key in image_keys:
            filename = os.path.basename(key)

            # Receuil de la classe à partir du nom
            if "_dandelion" in filename:
                class_label = "dandelion"
            elif "_grass" in filename:
                class_label = "grass"
            else:
                logging.warning(f"Nom de fichier non conforme, ignoré : {filename}")
                continue

            # Nettoie le nom de fichier
            cleaned_filename = filename.replace(f"_{class_label}", "")
            new_key = f"raw/{class_label}/{cleaned_filename}"

            # On télécharge
            buffer = BytesIO()
            MINIO_CLIENT.download_fileobj(BUCKET, key, buffer)
            buffer.seek(0)

            # On dépose
            MINIO_CLIENT.upload_fileobj(buffer, BUCKET, new_key)
            logging.info(f"Image déplacée : {key} → {new_key}")

            # On supprime l'original
            MINIO_CLIENT.delete_object(Bucket=BUCKET, Key=key)

        logging.info("Mise à jour de minio terminée.")

    except Exception as e:
        logging.error(f"Erreur lors de la mise à jour de la base de données : {e}")
        raise