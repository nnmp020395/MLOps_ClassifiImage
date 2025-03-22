from airflow.hooks.base import BaseHook
from sqlalchemy import text, create_engine
import requests
import boto3
from io import BytesIO
import pandas as pd
import logging

# Configuration de la connexion à MySQL
sql_alchemy_conn = 'sqlite:////Users/fabreindira/airflow/airflow.db'
engine = create_engine(sql_alchemy_conn)


# Fonction pour lire les données de MySQL
def read_from_sqlite():
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
        logging.error(f"Erreur lors de la lecture de SQLite: {e}")
        return []

# Fonction pour mettre à jour l'URL S3 dans MySQL
def update_sqlite_with_s3_urls(s3_urls):
    try:
        with engine.connect() as connection:
            UPDATE_QUERY = """
            UPDATE plants_data SET url_s3 = :s3_url WHERE url_source = :url_source
            """
            with connection.begin():
                connection.execute(
                    text(UPDATE_QUERY),
                    [ {"s3_url": s3_url, "url_source": url_source} for url_source, s3_url in s3_urls ])
        logging.info(f"{len(s3_urls)} URLs mises à jour dans MySQL.")
    except Exception as e:
        logging.error(f"Erreur lors de la mise à jour SQLite: {e}")


# Fonction pour télécharger l'image et la stocker dans S3
def process_images(**kwargs):
    try:
        aws_hook = BaseHook.get_connection('aws_s3_phuong')
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_hook.login,
            aws_secret_access_key=aws_hook.password
        )
        bucket_name = 'image-dadelion-grass'

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
                logging.error(f"Failed to process {url_source}: {e}")

            # Mettre à jour MySQL avec les URLs S3
        if s3_urls:
            update_sqlite_with_s3_urls(s3_urls)
    except Exception as e:
        logging.exception(f"Erreur lors du téléchargement des images ou de la connexion")
