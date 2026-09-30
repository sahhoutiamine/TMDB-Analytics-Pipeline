import json
import logging

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, RAW_MOVIES_FILE

logger = logging.getLogger(__name__)

NUMERIC_COLS = ["runtime", "budget", "revenue", "popularity", "vote_average", "vote_count"]
CATEGORICAL_COLS = ["original_language"]
TEXT_COLS = ["title", "overview"]
LIST_COLS = ["genres", "keywords"]
DATE_COLS = ["release_date"]


def read_one_movie(raw):
    return {
        "movie_id": raw.get("id"),
        "title": raw.get("title"),
        "overview": raw.get("overview"),
        "release_date": raw.get("release_date"),
        "runtime": raw.get("runtime"),
        "original_language": raw.get("original_language"),
        "genres": [g["name"] for g in raw.get("genres", [])],
        "keywords": [k["name"] for k in raw.get("keywords", {}).get("keywords", [])],
        "budget": raw.get("budget"),
        "revenue": raw.get("revenue"),
        "popularity": raw.get("popularity"),
        "vote_average": raw.get("vote_average"),
        "vote_count": raw.get("vote_count"),
    }


def load_raw():
    with open(RAW_MOVIES_FILE, "r", encoding="utf-8") as f:
        movie_list = json.load(f)
    return pd.DataFrame([read_one_movie(m) for m in movie_list])


def analyze_structure(df):
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nColumn types:")
    print(df.dtypes)
    print("\nFirst 3 rows:")
    print(df.head(3))


def report_missing(df):
    n_missing = df.isna().sum()
    pct_missing = (n_missing / len(df) * 100).round(1)
    report = pd.DataFrame({"n_missing": n_missing, "pct_missing": pct_missing})
    return report[report["n_missing"] > 0].sort_values("n_missing", ascending=False)


def report_duplicates(df):
    id_dupes = df["movie_id"].duplicated().sum()
    title_dupes = df["title"].duplicated().sum()
    return {"duplicate_movie_id": int(id_dupes), "duplicate_title": int(title_dupes)}


def drop_duplicates(df):
    before = len(df)
    df = df.drop_duplicates(subset="movie_id", keep="first").copy()
    removed = before - len(df)
    if removed:
        logger.info("Removed %s duplicate rows (same movie_id)", removed)
    return df


def detect_inconsistencies(df):
    today = pd.Timestamp.today()
    report = {
        "budget_zero": int((df["budget"] == 0).sum()),
        "revenue_zero": int((df["revenue"] == 0).sum()),
        "runtime_zero_or_null": int(((df["runtime"] == 0) | df["runtime"].isna()).sum()),
        "overview_empty": int((df["overview"].fillna("") == "").sum()),
        "vote_average_out_of_range": int(((df["vote_average"] < 0) | (df["vote_average"] > 10)).sum()),
        "release_date_in_future": int((pd.to_datetime(df["release_date"], errors="coerce") > today).sum()),
        "release_date_missing_or_invalid": int(pd.to_datetime(df["release_date"], errors="coerce").isna().sum()),
    }
    return report


def fix_inconsistencies(df):
    df = df.copy()
    df["budget"] = df["budget"].replace(0, np.nan)
    df["revenue"] = df["revenue"].replace(0, np.nan)
    df["runtime"] = df["runtime"].replace(0, np.nan)

    df.loc[(df["vote_average"] < 0) | (df["vote_average"] > 10), "vote_average"] = np.nan

    before = len(df)
    df = df[df["title"].notna() & (df["title"].str.strip() != "")]
    removed = before - len(df)
    if removed:
        logger.info("Removed %s rows without a title", removed)

    return df


def convert_dates(df):
    df = df.copy()
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    return df


def split_columns(df):
    groups = {
        "numeric": [c for c in NUMERIC_COLS if c in df.columns],
        "categorical": [c for c in CATEGORICAL_COLS if c in df.columns],
        "text": [c for c in TEXT_COLS if c in df.columns],
        "list": [c for c in LIST_COLS if c in df.columns],
        "date": [c for c in DATE_COLS if c in df.columns],
    }
    return groups


def clean(df):
    df = drop_duplicates(df)
    df = fix_inconsistencies(df)
    df = convert_dates(df)
    df["overview"] = df["overview"].fillna("")
    return df


def run_cleaning():
    logger.info("Loading raw data...")
    df_raw = load_raw()

    logger.info("Structure of the raw data:")
    analyze_structure(df_raw)

    print("\nMissing values (raw data):")
    print(report_missing(df_raw))

    print("\nDuplicates (raw data):", report_duplicates(df_raw))

    print("\nInconsistencies detected (raw data):")
    print(detect_inconsistencies(df_raw))

    logger.info("Cleaning...")
    df_clean = clean(df_raw)

    print("\nMissing values AFTER cleaning:")
    print(report_missing(df_clean))

    groups = split_columns(df_clean)
    print("\nColumn groups:", groups)

    df_clean.to_pickle(DATA_PROCESSED / "movies_clean.pkl")
    logger.info("Cleaned dataset saved: %s -> %s rows, %s columns",
                DATA_PROCESSED / "movies_clean.pkl", *df_clean.shape)
    return df_clean


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_cleaning()
