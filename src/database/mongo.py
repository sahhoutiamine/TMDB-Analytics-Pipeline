"""
MongoDB storage  (STEP 2)
=========================
Run it with:   python -m src.database.mongo

This module:
  1. Connects to MongoDB
  2. Empties the collection (so re-running never creates duplicates)
  3. Inserts every cleaned movie as one MongoDB "document"
  4. Creates an index on movie_id, so lookups by id are fast and stay unique
"""
import logging

import pandas as pd
from pymongo import MongoClient

from src.config import DATA_PROCESSED, MONGO_COLLECTION, MONGO_DB, MONGO_URI

logger = logging.getLogger(__name__)


def get_client():
    """One MongoClient, reused everywhere. `serverSelectionTimeoutMS` makes
    a wrong MONGO_URI fail fast (5s) instead of hanging forever."""
    return MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)


def get_collection():
    """Shortcut: client -> database -> collection (a "collection" in MongoDB
    is roughly the same idea as a table in SQL)."""
    return get_client()[MONGO_DB][MONGO_COLLECTION]


def dataframe_to_documents(df):
    """
    Turn a pandas DataFrame into a list of dicts that MongoDB can store.
    Pandas uses NaN for "missing"; MongoDB/JSON prefer `None` (null).
    `.where(df.notna(), None)` replaces every NaN with None.
    """
    return df.astype(object).where(df.notna(), None).to_dict("records")


def load_to_mongo(df=None):
    """Load the cleaned dataset into MongoDB. Returns the number of documents inserted."""
    if df is None:
        df = pd.read_pickle(DATA_PROCESSED / "movies_clean.pkl")

    collection = get_collection()
    collection.delete_many({})  # start from an empty collection every time

    documents = dataframe_to_documents(df)
    if documents:
        collection.insert_many(documents)

    # unique=True means: MongoDB itself refuses to insert two movies with the same id
    collection.create_index("movie_id", unique=True)

    logger.info("Inserted %s movies into MongoDB (%s.%s)", len(documents), MONGO_DB, MONGO_COLLECTION)
    return len(documents)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    n = load_to_mongo()
    print(f"{n} movies inserted into MongoDB.")
