from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Ajouter le dossier `scripts/` au PYTHONPATH pour importer les modules
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))

# Importer les fonctions des scripts Python
from download_from_db import get_image_urls, download_images
from update_s3_path import update_s3_paths
from prepare_data import clean_data
from train_model import train_pytorch_model

# Définition des paramètres du DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 3, 18),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Création du DAG
dag = DAG(
    "mlops_pipeline",
    default_args=default_args,
    description="Pipeline Airflow pour le projet MLOps",
    schedule=None,  
    catchup=False,
)
# Chemin de la base de données SQLite
DB_PATH = "../mlops_images.db"

# Définition des fonctions appelées par PythonOperator
def download_images_fct():
    BASE_SAVE_PATH = "../dataset" # Destination
    image_data = get_image_urls(DB_PATH)
    download_images(image_data)

def update_s3():
    update_s3_paths(DB_PATH)

def prepare_dataset():
    clean_data(DB_PATH)

def train_model():
    train_pytorch_model()

# Définition des tâches avec `PythonOperator`
download_task = PythonOperator(
    task_id="download_images",
    python_callable=download_images_fct,
    dag=dag,
)

update_s3_task = PythonOperator(
    task_id="update_s3_path",
    python_callable=update_s3,
    dag=dag,
)

prepare_data_task = PythonOperator(
    task_id="prepare_data",
    python_callable=prepare_dataset,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id="train_model",
    python_callable=train_model,
    dag=dag,
)

# Dépendances entre les tâches
download_task >> update_s3_task >> prepare_data_task >> train_model_task
