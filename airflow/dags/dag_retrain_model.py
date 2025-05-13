"""
DAG Airflow pour entraîner le modèle DINOv2 avec MLflow :
- Lancé une première fois au démarrage de l'infra (déploiement),
- Puis exécuté chaque samedi à minuit **uniquement s'il y a au moins 10 nouvelles images dans MinIO**.
"""

from datetime import datetime, timedelta

from airflow.operators.python import PythonOperator, ShortCircuitOperator
from utils.train_tasks import check_minio_image_count
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from airflow import DAG

# ------------------ CONFIGURATION DES PARAMÈTRES PAR DÉFAUT ------------------
default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# ------------------ DÉFINITION DU DAG ------------------
with DAG(
    dag_id="dinov2_REtrain_pipeline",
    default_args=default_args,
    description="Entraînement conditionnel hebdomadaire du modèle DINOv2 avec MLflow",
    schedule_interval="0 0 * * 6",  # Chaque samedi à 00:00
    start_date=datetime(2024, 4, 10),
    catchup=False,
    is_paused_upon_creation=False, 
) as dag:
    # -------- VÉRIFICATION DES NOUVELLES IMAGES ---------
    check_image = ShortCircuitOperator(  # Ne lance que si la condition est vraie
        task_id="Vérification_nouvelles_images",
        python_callable=check_minio_image_count,
        doc="Vérifie si au moins 10 nouvelles images sont présentes dans MinIO.",
    )
    # ------------------ MLflow TRAINING ------------------
    train_model = TriggerDagRunOperator(
        task_id="Entrainement_du_modele",
        trigger_dag_id="dinov2_train_pipeline",
        conf={},  
        wait_for_completion=False 
    )
    # ------------------ ORDRE DES TÂCHES ------------------
    check_image >> train_model
