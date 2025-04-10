from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import logging

# Configuration des logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

default_args = {
    'owner': 'mlops',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='dinov2_retrain_pipeline',
    default_args=default_args,
    description='Pipeline de réentraînement du modèle DinoV2',
    schedule_interval=None,
    start_date=datetime(2024, 4, 10),
    catchup=False
) as dag:

    def fetch_new_data():
        """ Vérification des nouvelles données dans S3 """
        logger.info("Étape 1 : Vérification des nouvelles données dans S3.")

    def retrain_model():
        """ Appelle du script d'entraînement existant """
        logger.info("Étape 2 : Début du réentraînement du modèle.")
        exit_code = os.system("python retrain_script.py")
        if exit_code != 0:
            logger.error("Erreur lors du réentraînement du modèle.")
        else:
            logger.info("Réentraînement terminé avec succès.")

    def upload_model_to_s3():
        """ Upload du modèle sur S3 """
        import boto3
        logger.info("Étape 3 : Début de l’upload du modèle sur S3.")
        s3 = boto3.client('s3')
        try:
            s3.upload_file("dinov2_classifier.pth", "image-dadelion-grass", "models/dinov2_classifier.pth")
            logger.info("Modèle uploadé avec succès sur S3.")
        except Exception as e:
            logger.error(f"Échec de l'upload du modèle : {e}")

    def redeploy_model():
        """ Redéploiement du modèle """
        logger.info("Étape 4 : Redéploiement du modèle.")
        # Tu peux intégrer ici un appel à un service, un webhook, etc.
        logger.info("Le modèle a été redéployé.")

    t1 = PythonOperator(task_id='fetch_new_data', python_callable=fetch_new_data)
    t2 = PythonOperator(task_id='retrain_model', python_callable=retrain_model)
    t3 = PythonOperator(task_id='upload_model_to_s3', python_callable=upload_model_to_s3)
    t4 = PythonOperator(task_id='redeploy_model', python_callable=redeploy_model)

    t1 >> t2 >> t3 >> t4
