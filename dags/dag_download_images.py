from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
#from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
#from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.hooks.base import BaseHook
from sqlalchemy import create_engine#, text
#import requests
#import boto3
#from io import BytesIO
#import pandas as pd
import logging
from lib.create_table_from_url import url_to_sql
from lib.store_images import process_images


# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

conn_aws = BaseHook.get_connection('aws_s3_phuong')
aws_access_key_id = conn_aws.login
aws_secret_access_key = conn_aws.password

# Configuration de la connexion à MySQL
sql_alchemy_conn = 'sqlite:////Users/fabreindira/airflow/airflow.db'
engine = create_engine(sql_alchemy_conn)


# # Fonction pour insérer les URLs dans MySQL
# def url_to_sql(**kwargs):
#     with engine.connect() as connection:
#         try:
#             CREATE_TABLE_QUERY = """
#             CREATE TABLE IF NOT EXISTS plants_data (
#                 id INT AUTO_INCREMENT PRIMARY KEY,
#                 url_source VARCHAR(255) NOT NULL UNIQUE,
#                 url_s3 VARCHAR(255) DEFAULT NULL,
#                 label VARCHAR(50) NOT NULL
#             );
#             """
#             connection.execute(text(CREATE_TABLE_QUERY))
#             logging.info("Table `plants_data` vérifiée/créée.")

#             base_url = "https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data"
#             labels = ["dandelion", "grass"]
#             data_to_insert = [{"url_source": f"{base_url}/{label}/{i:08d}.jpg", "url_s3": None, "label": label}
#                 for label in labels for i in range(200)
#             ]
#             INSERT_QUERY = """
#             INSERT OR IGNORE INTO plants_data (url_source, url_s3, label)
#             VALUES (:url_source, :url_s3, :label)
#             """

#             with connection.begin():
#                 connection.execute(text(INSERT_QUERY), data_to_insert)
            
#             logging.info(f"{len(data_to_insert)} enregistrements insérés avec succès.")
#             return len(data_to_insert)

#         except Exception as e:
#             logging.exception("Erreur MySQL lors de l'insertion des données")
#             connection.rollback()
#             return 0


# # Fonction pour lire les données de MySQL
# def read_from_sqlite():
#     try:
#         with engine.connect() as conn:
#             QUERY = """
#             SELECT url_source, label 
#             FROM plants_data 
#             WHERE url_s3 IS NULL
#             """
#             df = pd.read_sql(sql=QUERY, con=conn.connection)
#         return df[['url_source', 'label']].values.tolist()
#     except Exception as e:
#         logging.error(f"Erreur lors de la lecture de SQLite: {e}")
#         return []

# # Fonction pour mettre à jour l'URL S3 dans MySQL
# def update_sqlite_with_s3_urls(s3_urls):
#     try:
#         with engine.connect() as connection:
#             UPDATE_QUERY = """
#             UPDATE plants_data SET url_s3 = :s3_url WHERE url_source = :url_source
#             """
#             with connection.begin():
#                 connection.execute(
#                     text(UPDATE_QUERY),
#                     [ {"s3_url": s3_url, "url_source": url_source} for url_source, s3_url in s3_urls ])
#         logging.info(f"{len(s3_urls)} URLs mises à jour dans MySQL.")
#     except Exception as e:
#         logging.error(f"Erreur lors de la mise à jour SQLite: {e}")



with DAG(
    'mlops_project_pipeline_sqlite',
    default_args={
        'depends_on_past': False,
        'email': ['airflow@example.com'],
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 2,
        'retry_delay': timedelta(minutes=5),
    },
    description='MLOps pipeline for classification model',
    schedule_interval=None,  # declenchement manuel uniquement
    start_date=datetime(2025, 2, 19),
    catchup=False,
    tags=['example'],
) as dag:
    dag.doc_md = """
    MLOps pipeline for classification model.
    """

    # Tâche du DAG
    start_task = EmptyOperator(task_id='start_task')

    insert_urls_task = PythonOperator(
        task_id='insert_urls_to_sqlite',
        python_callable=url_to_sql,
        provide_context=True,
        dag=dag,
    )

    process_images_task = PythonOperator(
        task_id='process_images',
        python_callable=process_images,
        provide_context=True,
        dag=dag,
    )

    # Définir l'ordre des tâches
    start_task >> insert_urls_task >> process_images_task