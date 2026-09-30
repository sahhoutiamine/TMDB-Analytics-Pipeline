import logging

import pandas as pd
from pymongo import MongoClient

from src.config import DATA_PROCESSED, MONGO_COLLECTION, MONGO_DB, MONGO_URI

logger = logging.getLogger(__name__)


def get_client():
    return MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)


def get_collection():
    return get_client()[MONGO_DB][MONGO_COLLECTION]


def dataframe_to_documents(df):
    return df.astype(object).where(df.notna(), None).to_dict("records")


def load_to_mongo(df=None):
    if df is None:
        df = pd.read_pickle(DATA_PROCESSED / "movies_clean.pkl")

    collection = get_collection()
    collection.delete_many({})

    documents = dataframe_to_documents(df)
    if documents:
        collection.insert_many(documents)

    collection.create_index("movie_id", unique=True)

    logger.info("Inserted %s movies into MongoDB (%s.%s)", len(documents), MONGO_DB, MONGO_COLLECTION)
    return len(documents)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    n = load_to_mongo()
    print(f"{n} movies inserted into MongoDB.")
