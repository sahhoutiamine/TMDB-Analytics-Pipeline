import logging

from src.cleaning.clean import run_cleaning
from src.database.mongo import load_to_mongo
from src.extraction.extract import run_extraction
from src.features.engineering import run_features

logging.basicConfig(level=logging.INFO)


def run_ml():
    pass


if __name__ == "__main__":
    run_extraction()
    run_cleaning()
    run_features()
    load_to_mongo()
    run_ml()
