#from airflow.hooks.base import BaseHook
from sqlalchemy import text, create_engine
import requests
import boto3
from io import BytesIO
import pandas as pd
import logging
import os

# Configuration de la connexion à MySQL
sql_alchemy_conn = 'postgresql+psycopg2://airflow:airflow@postgres/airflow'
engine = create_engine(sql_alchemy_conn)

# Récupérer les identifiants AWS à partir des variables d'environnement
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_default_region = os.getenv('AWS_DEFAULT_REGION')
bucket_name = 'image-dadelion-grass'

# Fonction pour lire les données de MySQL
def read_from_postgresql():
    try:
        with engine.connect() as conn:
            QUERY = """
            SELECT url_source, label 
            FROM plants_data 
            WHERE url_s3 IS NULL
            """
            df = pd.read_sql(sql=QUERY, con=conn.connection)
        return df[['url_source', 'label']].values.tolist()
    except Exception as e:
        logging.error(f"Erreur lors de la lecture de Postgresql: {e}")
        return []

# Fonction pour mettre à jour l'URL S3 dans MySQL
def update_postgresql_with_s3_urls(s3_urls):
    try:
        with engine.connect() as connection:
            UPDATE_QUERY = """
            UPDATE plants_data SET url_s3 = :s3_url WHERE url_source = :url_source
            """
            with connection.begin():
                connection.execute(
                    text(UPDATE_QUERY),
                    [ {"s3_url": s3_url, "url_source": url_source} for url_source, s3_url in s3_urls ])
        logging.info(f"{len(s3_urls)} URLs mises à jour dans Postgresql.")
    except Exception as e:
        logging.error(f"Erreur lors de la mise à jour Postgresql: {e}")


# Fonction pour télécharger l'image et la stocker dans S3
def process_images(**kwargs):
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_default_region
        )

        rows = read_from_postgresql()
        s3_urls = []

        for url_source, label in rows:
            try:
                response = requests.get(url_source)
                image_data = BytesIO(response.content)
                s3_key = f"raw/{label}/{url_source.split('/')[-1]}"
                s3_client.upload_fileobj(image_data, bucket_name, s3_key)
                s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
                s3_urls.append((url_source, s3_url))
            except Exception as e:
                logging.error(f"Failed to process {url_source}: {e}")

            # Mettre à jour MySQL avec les URLs S3
        if s3_urls:
            update_postgresql_with_s3_urls(s3_urls)
    except Exception as e:
        logging.exception(f"Erreur lors du téléchargement des images ou de la connexion")
