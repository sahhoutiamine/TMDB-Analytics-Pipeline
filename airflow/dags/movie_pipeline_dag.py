"""Extraction -> Cleaning -> Features -> MongoDB -> ML"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {"owner": "data-team", "retries": 2, "retry_delay": timedelta(minutes=2)}


def _extract():
    from src.extraction.extract import run_extraction
    run_extraction()

def _clean():
    from src.cleaning.clean import run_cleaning
    run_cleaning()

def _features():
    from src.features.engineering import run_features
    run_features()

def _mongo():
    from src.database.mongo import load_to_mongo
    load_to_mongo()

def _ml():
    from src.pipeline import run_ml
    run_ml()


with DAG("movie_ml_pipeline", default_args=default_args, start_date=datetime(2025, 1, 1),
         schedule="@weekly", catchup=False, tags=["movies", "ml"]) as dag:
    t1 = PythonOperator(task_id="extract", python_callable=_extract)
    t2 = PythonOperator(task_id="clean", python_callable=_clean)
    t3 = PythonOperator(task_id="features", python_callable=_features)
    t4 = PythonOperator(task_id="mongodb", python_callable=_mongo)
    t5 = PythonOperator(task_id="ml", python_callable=_ml)
    t1 >> t2 >> t3 >> t4 >> t5
