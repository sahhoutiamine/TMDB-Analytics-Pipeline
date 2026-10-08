from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

from src.extraction.extract import run_extraction
from src.cleaning.clean import run_cleaning
from src.features.engineering import run_features
from src.database.mongo import load_to_mongo
from src.models.classification import run_classification

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'tmdb_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False
) as dag:

    extract_task = PythonOperator(
        task_id='extraction',
        python_callable=run_extraction
    )

    clean_task = PythonOperator(
        task_id='nettoyage',
        python_callable=run_cleaning
    )

    features_task = PythonOperator(
        task_id='features',
        python_callable=run_features
    )

    mongo_task = PythonOperator(
        task_id='mongodb',
        python_callable=load_to_mongo
    )

    ml_task = PythonOperator(
        task_id='ml',
        python_callable=run_classification
    )

    extract_task >> clean_task >> features_task >> mongo_task >> ml_task
