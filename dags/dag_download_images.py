"""
DAG pour télécharger et traiter des images.

Ce fichier définit un DAG Airflow qui exécute les étapes suivantes :
- Récupération des URLs d'images depuis une source externe.
- Insertion des URLs dans une base de données SQLite.
- Téléchargement et traitement des images depuis les URLs stockées.
"""
import logging
from datetime import datetime, timedelta

from airflow.hooks.base import BaseHook
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from lib.create_table_from_url import url_to_sql
from lib.store_images import process_images
from sqlalchemy import create_engine

from airflow import DAG

# ------------------ CONFIGURATION LOGGING ------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------ ACCÈS AUX IDENTIFIANTS AWS ------------------
# Récupération des identifiants depuis la connexion Airflow (préconfigurée)
conn_aws = BaseHook.get_connection("aws_s3_phuong")
aws_access_key_id = conn_aws.login
aws_secret_access_key = conn_aws.password

# ------------------ CONFIGURATION BASE DE DONNÉES ------------------
# Connexion vers une base SQLite locale pour stocker les URLs et métadonnées
sql_alchemy_conn = "sqlite:////Users/fabreindira/airflow/airflow.db"
engine = create_engine(sql_alchemy_conn)

# ------------------ DÉFINITION DU DAG ------------------
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
    schedule_interval=None,  # déclenchement manuel uniquement
    start_date=datetime(2025, 2, 19),
    catchup=False,
    tags=["example"],
) as dag:
    dag.doc_md = """
    Ce pipeline Airflow exécute une suite de tâches pour un projet MLOps :
    - insertion des URLs d'images dans une base de données SQLite,
    - téléchargement et traitement des images depuis les URLs stockées.
    """

    # ------------------ TÂCHE DE DÉPART ------------------
    start_task = EmptyOperator(
        task_id="start_task",
        doc_md="Tâche de départ vide, sert de point d'entrée dans le DAG.",
    )

    # ------------------ INSERTION DES URLS DANS SQLITE ------------------
    insert_urls_task = PythonOperator(
        task_id="insert_urls_to_sqlite",
        python_callable=url_to_sql,
        provide_context=True,
        doc_md="""
        Appelle la fonction `url_to_sql` pour insérer des URLs d’images
        dans la base SQLite à partir d’un fichier ou d’une source externe.
        """,
    )

    # ------------------ TÉLÉCHARGEMENT & PRÉTRAITEMENT DES IMAGES ------------------
    process_images_task = PythonOperator(
        task_id="process_images",
        python_callable=process_images,
        provide_context=True,
        doc_md="""
        Tâche qui télécharge les images à partir des URLs stockées en base,
        les traite (resize, conversion RGB...) et les stocke dans un répertoire cible.
        """,
    )

    # ------------------ CHAÎNAGE DES TÂCHES ------------------
    start_task >> insert_urls_task >> process_images_task
