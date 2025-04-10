from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import subprocess
import logging

default_args = {
    'owner': 'mlops',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='dinov2_train_pipeline_with_mlflow',
    default_args=default_args,
    description='Pipeline complet avec traçage MLflow',
    schedule_interval=None,
    start_date=datetime(2024, 4, 10),
    catchup=False
) as dag:

    def run_mlflow_tracking():
        logging.info("Lancement du script mlflow_tracking.py via subprocess.")
        try:
            result = subprocess.run(
                ["python", "/opt/airflow/src/mlflow_tracking.py"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            logging.info("Script exécuté avec succès.")
            logging.info("STDOUT:\n" + result.stdout)
            logging.info("STDERR:\n" + result.stderr)
        except subprocess.CalledProcessError as e:
            logging.error("Échec du script mlflow_tracking.py.")
            logging.error("STDOUT:\n" + e.stdout)
            logging.error("STDERR:\n" + e.stderr)
            raise e

    # Tâche :
    t_mlflow_tracking = PythonOperator(
        task_id='mlflow_tracking',
        python_callable=run_mlflow_tracking
    )

    # Lancement : 
    t_mlflow_tracking
