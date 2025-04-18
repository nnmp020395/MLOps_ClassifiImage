"""
DAG pour télécharger et traiter des images.

Ce fichier définit un DAG Airflow qui exécute les étapes suivantes :
- Récupération des URLs d'images depuis une source externe.
- Insertion des URLs dans une base de données SQLite.
- Téléchargement et traitement des images depuis les URLs stockées.
"""
import logging
import os
from datetime import datetime, timedelta

from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from sqlalchemy import create_engine  # , text
from utils.create_table_from_url import url_to_sql
from utils.store_images import process_images

from airflow import DAG

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_default_region = os.getenv("AWS_DEFAULT_REGION")

# Configuration de la connexion à MySQL
sql_alchemy_conn = "postgresql+psycopg2://airflow:airflow@postgres/airflow"
engine = create_engine(sql_alchemy_conn)


with DAG(
    "mlops_project_pipeline_sqlite",
    default_args={
        "depends_on_past": False,
        "email": ["airflow@example.com"],
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    },
    description="MLOps pipeline for classification model",
    schedule_interval=None,  # declenchement manuel uniquement
    start_date=datetime(2025, 2, 19),
    catchup=False,
    tags=["example"],
) as dag:
    dag.doc_md = """
    MLOps pipeline for classification model.
    """

    # Tâche du DAG
    start_task = EmptyOperator(task_id="start_task")

    insert_urls_task = PythonOperator(
        task_id="insert_urls_to_postgresql",
        python_callable=url_to_sql,
        provide_context=True,
        dag=dag,
    )

    process_images_task = PythonOperator(
        task_id="process_images",
        python_callable=process_images,
        provide_context=True,
        dag=dag,
    )

    # Définir l'ordre des tâches
    start_task >> insert_urls_task >> process_images_task
