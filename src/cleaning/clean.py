"""Step 2: load raw JSON -> tidy DataFrame -> cleaned parquet/csv."""
import json
import logging

import pandas as pd

from src.config import DATA_PROCESSED, DATA_RAW

logger = logging.getLogger(__name__)
KEEP = ["id", "title", "overview", "release_date", "runtime", "original_language",
        "budget", "revenue", "popularity", "vote_average", "vote_count"]


def load_raw() -> pd.DataFrame:
    rows = []
    for f in sorted(DATA_RAW.glob("movie_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        row = {k: d.get(k) for k in KEEP}
        row["movie_id"] = row.pop("id")
        row["genres"] = [g["name"] for g in d.get("genres", [])]
        row["keywords"] = [k["name"] for k in d.get("keywords", {}).get("keywords", [])]
        rows.append(row)
    return pd.DataFrame(rows)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # TODO: types, duplicates, missing values, inconsistencies (budget/revenue/runtime == 0),
    #       date conversion, split numeric / categorical / text columns
    df = df.drop_duplicates(subset="movie_id").copy()
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    return df


def run_cleaning() -> pd.DataFrame:
    df = clean(load_raw())
    df.to_pickle(DATA_PROCESSED / "movies_clean.pkl")  # pickle keeps list columns
    logger.info("Cleaned dataset: %s", df.shape)
    return df
