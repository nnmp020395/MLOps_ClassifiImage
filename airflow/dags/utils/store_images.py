"""
Module pour gérer le stockage des images sur S3 et la mise à jour de la base\
SQLite.

Ce fichier contient des fonctions pour :
1. Lire les URLs d'images depuis une base SQLite.
2. Télécharger les images et les uploader sur un bucket S3.
3. Mettre à jour la base SQLite avec les URLs des images stockées sur S3.
"""
# from airflow.hooks.base import BaseHook
import logging
import os
from io import BytesIO

import boto3
import pandas as pd
import requests
from sqlalchemy import create_engine, text

# Configuration de la connexion à MySQL
sql_alchemy_conn = "postgresql+psycopg2://airflow:airflow@postgres/airflow"
engine = create_engine(sql_alchemy_conn)

# Récupérer les identifiants AWS à partir des variables d'environnement
minio_endpoint = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
minio_access_key = os.getenv("AWS_ACCESS_KEY_ID") 
minio_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
bucket_name = "image-dandelion-grass"
region_name = os.getenv("AWS_DEFAULT_REGION", "us-east-1")


# Fonction pour lire les données de MySQL
def read_from_postgresql():
    """
    Lit les lignes de la base SQLite dont le champ `url_s3` est NULL.

    Retourne :
    ---------
    list of tuples :
        Liste contenant les paires (url_source, label) pour les images non \
            encore stockées sur S3.
    """
    try:
        with engine.connect() as conn:
            QUERY = """
            SELECT url_source, label
            FROM plants_data
            WHERE url_s3 IS NULL
            """
            df = pd.read_sql(sql=QUERY, con=conn.connection)
        return df[["url_source", "label"]].values.tolist()
    except Exception as e:
        logging.error(f"Erreur lors de la lecture de Postgresql: {e}")
        return []


def update_postgresql_with_s3_urls(s3_urls):
    """
    Met à jour les lignes dans Postgresql avec les URLs S3 associées.

    Paramètres :
    -----------
    s3_urls : list of tuples
        Liste de tuples (url_source, s3_url) à mettre à jour dans la base Postgresql.
    """
    try:
        with engine.connect() as connection:
            UPDATE_QUERY = """
            UPDATE plants_data SET url_s3 = :s3_url WHERE url_source = :url_source
            """
            with connection.begin():
                connection.execute(
                    text(UPDATE_QUERY),
                    [
                        {"s3_url": s3_url, "url_source": url_source}
                        for url_source, s3_url in s3_urls
                    ],
                )
        logging.info(f"{len(s3_urls)} URLs mises à jour dans Postgresql.")
    except Exception as e:
        logging.error(f"Erreur lors de la mise à jour Postgresql: {e}")


# Fonction pour télécharger l'image et la stocker dans S3
def process_images(**kwargs):
    """
    Process réalisé pour stocker les images.

    Tâche Airflow qui :
    1. Télécharge les images depuis leurs URLs source.
    2. Les stocke dans un bucket S3.
    3. Met à jour la base SQLite avec les URLs S3 correspondantes.

    Paramètres :
    -----------
    kwargs : dict
        Contexte de la tâche Airflow (non utilisé ici).
    """
    try:
        s3_client = boto3.client(
            "s3",
            endpoint_url=minio_endpoint,
            aws_access_key_id=minio_access_key,
            aws_secret_access_key=minio_secret_key,
            region_name=region_name,
            use_ssl=False,
        )

        # Vérifier si le bucket existe déjà
        existing_buckets = [b['Name'] for b in s3_client.list_buckets()['Buckets']]
        if bucket_name not in existing_buckets:
            s3_client.create_bucket(Bucket=bucket_name)
            logging.info(f"Bucket '{bucket_name}' created.")

        rows = read_from_postgresql()
        s3_urls = []

        for url_source, label in rows:
            try:
                response = requests.get(url_source)
                image_data = BytesIO(response.content)
                s3_key = f"raw/{label}/{url_source.split('/')[-1]}"
                s3_client.upload_fileobj(image_data, bucket_name, s3_key)

                # Construct MinIO access URL
                s3_url = f"{minio_endpoint}/{bucket_name}/{s3_key}"
                s3_urls.append((url_source, s3_url))

            except Exception as e:
                logging.error(f"Failed to process {url_source}: {e}")

            # Mettre à jour MySQL avec les URLs S3
        if s3_urls:
            update_postgresql_with_s3_urls(s3_urls)
    except Exception as e:
        logging.exception(
            f"Erreur lors du téléchargement des images ou de la connexion : {e}"
        )
