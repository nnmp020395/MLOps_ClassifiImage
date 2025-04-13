"""
Module pour gérer le stockage des images sur S3 et la mise à jour de la base\
SQLite.

Ce fichier contient des fonctions pour :
1. Lire les URLs d'images depuis une base SQLite.
2. Télécharger les images et les uploader sur un bucket S3.
3. Mettre à jour la base SQLite avec les URLs des images stockées sur S3.
"""
import logging
from io import BytesIO

import boto3
import pandas as pd
import requests
from airflow.hooks.base import BaseHook
from sqlalchemy import create_engine, text

# ------------------ CONFIGURATION ------------------

# Configuration de la connexion à la base SQLite via SQLAlchemy
sql_alchemy_conn = "sqlite:////Users/fabreindira/airflow/airflow.db"
engine = create_engine(sql_alchemy_conn)


def read_from_sqlite():
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
        logging.error(f"Erreur lors de la lecture de SQLite: {e}")
        return []


def update_sqlite_with_s3_urls(s3_urls):
    """
    Met à jour les lignes dans SQLite avec les URLs S3 associées.

    Paramètres :
    -----------
    s3_urls : list of tuples
        Liste de tuples (url_source, s3_url) à mettre à jour dans la base SQLite.
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
        logging.info(f"{len(s3_urls)} URLs mises à jour dans SQLite.")
    except Exception as e:
        logging.error(f"Erreur lors de la mise à jour SQLite: {e}")


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
        aws_hook = BaseHook.get_connection("aws_s3_phuong")
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_hook.login,
            aws_secret_access_key=aws_hook.password,
        )
        bucket_name = "image-dadelion-grass"

        rows = read_from_sqlite()
        s3_urls = []

        for url_source, label in rows:
            try:
                response = requests.get(url_source)
                image_data = BytesIO(response.content)
                s3_key = f"{label}/{url_source.split('/')[-1]}"
                s3_client.upload_fileobj(image_data, bucket_name, s3_key)
                s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
                s3_urls.append((url_source, s3_url))
            except Exception as e:
                logging.error(f"Échec du traitement de {url_source} : {e}")

        # Mise à jour de SQLite si des URLs S3 ont été générées
        if s3_urls:
            update_sqlite_with_s3_urls(s3_urls)

    except Exception as e:
        logging.exception(
            f"Erreur lors du téléchargement des images ou de la connexion \
                à S3 : {e}"
        )
