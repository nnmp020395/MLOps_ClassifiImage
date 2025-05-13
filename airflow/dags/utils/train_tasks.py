"""
Module pour gérer l'entrainement et le ré-entrainement du modèle DINOv2.

Ce module contient les fonctions nécessaires pour :
1. Vérifier la présence de nouvelles images images dans le bucket MinIO.
2. Lancer le script de suivi d'expérience avec MLflow.
"""

import logging
import os

import boto3
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from io import BytesIO
from sqlalchemy import create_engine, text
from utils.store_images import update_postgresql_with_s3_urls



# ------------------ Configuration PostgreSQL ------------------
sql_alchemy_conn = "postgresql+psycopg2://airflow:airflow@postgres/airflow"
engine = create_engine(sql_alchemy_conn)

# ------------------ CONFIGURATION MINIO ------------------
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

# ------------------ FONCTIONS ------------------
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


def update_database_and_store_metadata():
    """
    Déplace les images validées vers leur dossier définitif (raw/dandelion ou raw/grass),
    les renomme, puis enregistre leur chemin dans PostgreSQL.
    """
    try:
        response = MINIO_CLIENT.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX_RAW)
        objects = response.get("Contents", [])
        image_keys = [obj["Key"] for obj in objects if obj["Key"].lower().endswith((".jpg", ".jpeg", ".png"))]

        new_images_to_store = []

        for key in image_keys:
            filename = os.path.basename(key)

            # Identifie le label et le retire du nom de l'image
            if "_dandelion" in filename:
                label = "dandelion"
            elif "_grass" in filename:
                label = "grass"
            else:
                logging.warning(f"Fichier ignoré (classe inconnue) : {filename}")
                continue

            cleaned_filename = filename.replace(f"_{label}", "")
            dest_key = f"raw/{label}/{cleaned_filename}"

            # Télécharge
            buffer = BytesIO()
            MINIO_CLIENT.download_fileobj(BUCKET, key, buffer)
            buffer.seek(0)

            # Dépose
            MINIO_CLIENT.upload_fileobj(buffer, BUCKET, dest_key)
            logging.info(f"Image déplacée : {key} → {dest_key}")

            # Supprime l'ancienne image
            MINIO_CLIENT.delete_object(Bucket=BUCKET, Key=key)

            # Build URL MinIO accessible
            s3_url = f"{os.getenv('MINIO_ENDPOINT', 'http://minio:9000')}/{BUCKET}/{dest_key}"
            new_images_to_store.append((cleaned_filename, label, s3_url))

        # Enregistrement dans PostgreSQL
        if new_images_to_store:
            update_postgresql_with_s3_urls(new_images_to_store)

    except Exception as e:
        logging.error(f"Erreur durant le traitement des images : {e}")
        raise