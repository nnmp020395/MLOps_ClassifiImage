from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Ajouter le dossier `scripts/` au PYTHONPATH pour importer le module
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))

# Importer la fonction principale du script
from store_urls_sqlite import url_to_sql

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
    "store_urls_pipeline",
    default_args=default_args,
    description="Pipeline Airflow pour stocker les URLs des images dans SQLite",
    schedule=None,  
    catchup=False,
)

# Fonction wrapper pour exécuter la tâche Airflow
def run_store_urls():
    db_path = "../mlops_images.db"  
    url_to_sql(db_path)

# Tâche pour exécuter le script via PythonOperator
store_urls = PythonOperator(
    task_id="store_urls",
    python_callable=run_store_urls,
    dag=dag,
)

store_urls
