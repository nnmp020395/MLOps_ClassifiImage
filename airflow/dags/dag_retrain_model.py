"""
DAG Airflow pour entraîner le modèle DINOv2 avec MLflow :
- Lancé une première fois au démarrage de l'infra (déploiement),
- Puis exécuté chaque samedi à minuit **uniquement s'il y a au moins 10 nouvelles images dans MinIO**.
"""

from datetime import datetime, timedelta

from airflow.operators.python import PythonOperator, ShortCircuitOperator
from utils.train_tasks import check_minio_image_count, run_mlflow_tracking

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
    catchup=True,  # Pour forcer l'exécution à la première planification
) as dag:
    # -------- VÉRIFICATION DES NOUVELLES IMAGES ---------
    check_image = ShortCircuitOperator(  # Ne se lance que si la condition est vraie
        task_id="Vérification_nouvelles_images",
        python_callable=check_minio_image_count,
        doc="Vérifie si au moins 10 nouvelles images sont présentes dans MinIO.",
    )
    # ------------------ MLflow TRAINING ------------------
    train_model = PythonOperator(
        task_id="Entrainement_du_modele",
        python_callable=run_mlflow_tracking,
        doc="Lance le script de suivi d'expérience avec MLflow.",
    )
    # ------------------ ORDRE DES TÂCHES ------------------
    check_image >> train_model
