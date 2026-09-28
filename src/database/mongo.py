"""Step 2: MongoDB storage."""
import pandas as pd
from pymongo import MongoClient

from src.config import DATA_PROCESSED, MONGO_COLLECTION, MONGO_DB, MONGO_URI


def get_collection():
    return MongoClient(MONGO_URI)[MONGO_DB][MONGO_COLLECTION]


def load_to_mongo(df: pd.DataFrame | None = None) -> int:
    if df is None:
        df = pd.read_pickle(DATA_PROCESSED / "movies_clean.pkl")
    col = get_collection()
    col.delete_many({})
    docs = df.astype(object).where(df.notna(), None).to_dict("records")
    if docs:
        col.insert_many(docs)
    col.create_index("movie_id", unique=True)
    return len(docs)
