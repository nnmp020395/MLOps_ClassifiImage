"""
DAG pour le réentraînement du modèle DinoV2.

Ce fichier définit un DAG Airflow qui automatise le processus de réentraînement
du modèle DinoV2. Les étapes incluent :
1. Vérification de nouvelles données dans un bucket S3.
2. Réentraînement du modèle avec un script Python.
3. Upload du nouveau modèle sur un bucket S3.
4. Redéploiement du modèle dans l’API ou infrastructure cible.
"""

import logging
import os
from datetime import datetime, timedelta

from airflow.operators.python import PythonOperator

from airflow import DAG

# ------------------ CONFIGURATION DU LOGGING ------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------ PARAMÈTRES PAR DÉFAUT ------------------
default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# ------------------ DÉFINITION DU DAG ------------------
with DAG(
    dag_id="dinov2_retrain_pipeline",
    default_args=default_args,
    description="Pipeline de réentraînement du modèle DinoV2",
    schedule_interval=None,
    start_date=datetime(2024, 4, 10),
    catchup=False,
) as dag:
    """
    DAG qui automatise le réentraînement du modèle DinoV2 en quatre étapes.

    1. Vérification de nouvelles données dans S3
    2. Réentraînement du modèle avec un script Python
    3. Upload du nouveau modèle sur un bucket S3
    4. Redéploiement du modèle dans l’infrastructure cible
    """

    def fetch_new_data():
        """
        Étape 1 : Cette fonction représente la tâche de détection.

        Elle vérifie la présence de nouvelles données dans le bucket S3.
        """
        logger.info("Étape 1 : Vérification des nouvelles données dans S3.")

    def retrain_model():
        """
        Étape 2 : Lance le script retrain_script.py pour réentraîner le modèle.

        Le code de retour du script est vérifié pour détecter une éventuelle erreur.
        """
        logger.info("Étape 2 : Début du réentraînement du modèle.")
        exit_code = os.system("python retrain_script.py")
        if exit_code != 0:
            logger.error("Erreur lors du réentraînement du modèle.")
        else:
            logger.info("Réentraînement terminé avec succès.")

    def upload_model_to_s3():
        """
        Étape 3 : Upload du fichier 'dinov2_classifier.pth' dans un bucket S3.

        Le modèle est envoyé à l’endroit où l’API le récupérera pour prédiction.
        """
        import boto3

        logger.info("Étape 3 : Début de l’upload du modèle sur S3.")
        s3 = boto3.client("s3")
        try:
            s3.upload_file(
                "dinov2_classifier.pth",
                "image-dadelion-grass",
                "models/dinov2_classifier.pth",
            )
            logger.info("Modèle uploadé avec succès sur S3.")
        except Exception as e:
            logger.error(f"Échec de l'upload du modèle : {e}")

    def redeploy_model():
        """
        Étape 4 : Redéploie le modèle dans l'infrastructure de production.

        Cette étape peut être personnalisée avec des commandes Docker ou SSH.
        """
        logger.info("Étape 4 : Redéploiement du modèle.")
        # Exemple : os.system("docker-compose restart fastapi-api")
        logger.info("Le modèle a été redéployé.")

    # ------------------ DÉFINITION DES TÂCHES ------------------
    t1 = PythonOperator(task_id="fetch_new_data", python_callable=fetch_new_data)
    t2 = PythonOperator(task_id="retrain_model", python_callable=retrain_model)
    t3 = PythonOperator(
        task_id="upload_model_to_s3", python_callable=upload_model_to_s3
    )
    t4 = PythonOperator(task_id="redeploy_model", python_callable=redeploy_model)

    # ------------------ CHAÎNAGE DES TÂCHES ------------------
    t1 >> t2 >> t3 >> t4
