from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
#from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
#from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.hooks.base import BaseHook
from sqlalchemy import create_engine#, text
#import requests
#import boto3
#from io import BytesIO
#import pandas as pd
import logging
from lib.create_table_from_url import url_to_sql
from lib.store_images import process_images


# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

conn_aws = BaseHook.get_connection('aws_s3_phuong')
aws_access_key_id = conn_aws.login
aws_secret_access_key = conn_aws.password

# Configuration de la connexion à MySQL
sql_alchemy_conn = 'sqlite:////Users/fabreindira/airflow/airflow.db'
engine = create_engine(sql_alchemy_conn)


with DAG(
    'mlops_project_pipeline_sqlite',
    default_args={
        'depends_on_past': False,
        'email': ['airflow@example.com'],
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 2,
        'retry_delay': timedelta(minutes=5),
    },
    description='MLOps pipeline for classification model',
    schedule_interval=None,  # declenchement manuel uniquement
    start_date=datetime(2025, 2, 19),
    catchup=False,
    tags=['example'],
) as dag:
    dag.doc_md = """
    MLOps pipeline for classification model.
    """

    # Tâche du DAG
    start_task = EmptyOperator(task_id='start_task')

    insert_urls_task = PythonOperator(
        task_id='insert_urls_to_sqlite',
        python_callable=url_to_sql,
        provide_context=True,
        dag=dag,
    )

    process_images_task = PythonOperator(
        task_id='process_images',
        python_callable=process_images,
        provide_context=True,
        dag=dag,
    )

    # Définir l'ordre des tâches
    start_task >> insert_urls_task >> process_images_task