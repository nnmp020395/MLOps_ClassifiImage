"""
DAG pour entraîner un modèle MLFlow.

Ce fichier définit un DAG Airflow qui exécute les étapes nécessaires pour
entraîner un modèle en utilisant MLFlow.
"""

import logging
import subprocess
from datetime import datetime, timedelta

from airflow.operators.python import PythonOperator

from airflow import DAG

# ------------------ CONFIGURATION DES PARAMÈTRES PAR DÉFAUT ------------------
default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# ------------------ DÉFINITION DU DAG ------------------
with DAG(
    dag_id="dinov2_train_pipeline_with_mlflow",
    default_args=default_args,
    description="Pipeline complet avec traçage MLflow",
    schedule_interval=None,
    start_date=datetime(2024, 4, 10),
    catchup=False,
) as dag:

    def run_mlflow_tracking():
        """
        Fonction exécutée par le PythonOperator.

        Elle lance le script 'mlflow_tracking.py' via un sous-processus,
        puis loggue la sortie standard et les erreurs.
        """
        logging.info("Lancement du script mlflow_tracking.py via subprocess.")
        try:
            result = subprocess.run(
                ["python", "/opt/airflow/src/mlflow_tracking.py"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            logging.info("Script exécuté avec succès.")
            logging.info("STDOUT:\n" + result.stdout)
            logging.info("STDERR:\n" + result.stderr)
        except subprocess.CalledProcessError as e:
            logging.error("Échec du script mlflow_tracking.py.")
            logging.error("STDOUT:\n" + e.stdout)
            logging.error("STDERR:\n" + e.stderr)
            raise e

    # ------------------ TÂCHE MLflow ------------------
    t_mlflow_tracking = PythonOperator(
        task_id="mlflow_tracking",
        python_callable=run_mlflow_tracking,
        doc="Lance le script de suivi d'expérience avec MLflow.",
    )

    # ------------------ DÉFINITION DES DÉPENDANCES ------------------
    t_mlflow_tracking
