"""
Cleaning script  (STEP 2)
==========================
Run it with:   python -m src.cleaning.clean

What it does, in order (each one is a bullet point of the project):
  1. Load the raw data and look at its types and structure
  2. Find missing values and duplicates
  3. Detect inconsistencies (values that don't make sense)
  4. Convert the date column to a real date type
  5. Split the columns into numeric / categorical / text / list groups
  6. Save the cleaned DataFrame to data/processed/

Nothing here is "magic": every function does ONE simple thing, and prints
what it found, so you can see the effect of each step.
"""
import json
import logging

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, RAW_MOVIES_FILE

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Column groups (used again later in EDA, feature engineering and modeling)
# --------------------------------------------------------------------------
NUMERIC_COLS = ["runtime", "budget", "revenue", "popularity", "vote_average", "vote_count"]
CATEGORICAL_COLS = ["original_language"]
TEXT_COLS = ["title", "overview"]
LIST_COLS = ["genres", "keywords"]           # columns that contain a list of values
DATE_COLS = ["release_date"]


# --------------------------------------------------------------------------
# 0. Load the raw JSON file into a DataFrame
# --------------------------------------------------------------------------
def read_one_movie(raw):
    """Take one movie (a dict from the JSON file) and keep only the 13 fields we need."""
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
    """Read data/raw/movies_raw.json and return a DataFrame with 13 columns."""
    with open(RAW_MOVIES_FILE, "r", encoding="utf-8") as f:
        movie_list = json.load(f)
    return pd.DataFrame([read_one_movie(m) for m in movie_list])


# --------------------------------------------------------------------------
# 1. Types and structure
# --------------------------------------------------------------------------
def analyze_structure(df):
    """
    Print basic information about the DataFrame:
    shape, column types, and a few example rows.
    This is always the FIRST thing to do with a new dataset.
    """
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nColumn types:")
    print(df.dtypes)
    print("\nFirst 3 rows:")
    print(df.head(3))


# --------------------------------------------------------------------------
# 2. Missing values and duplicates
# --------------------------------------------------------------------------
def report_missing(df):
    """Return a DataFrame with the number and % of missing values per column."""
    n_missing = df.isna().sum()
    pct_missing = (n_missing / len(df) * 100).round(1)
    report = pd.DataFrame({"n_missing": n_missing, "pct_missing": pct_missing})
    return report[report["n_missing"] > 0].sort_values("n_missing", ascending=False)


def report_duplicates(df):
    """
    Check for duplicates.
    - movie_id duplicated: should NEVER happen (each id is unique in TMDB) -> real problem.
    - title duplicated: can be normal (remakes, sequels sharing a name) -> just informative.
    """
    id_dupes = df["movie_id"].duplicated().sum()
    title_dupes = df["title"].duplicated().sum()
    return {"duplicate_movie_id": int(id_dupes), "duplicate_title": int(title_dupes)}


def drop_duplicates(df):
    """Remove rows that share the same movie_id, keeping the first one."""
    before = len(df)
    df = df.drop_duplicates(subset="movie_id", keep="first").copy()
    removed = before - len(df)
    if removed:
        logger.info("Removed %s duplicate rows (same movie_id)", removed)
    return df


# --------------------------------------------------------------------------
# 3. Inconsistencies
# --------------------------------------------------------------------------
def detect_inconsistencies(df):
    """
    Look for values that don't make sense and return a report (dict of counts).
    In TMDB, 0 in budget/revenue/runtime almost always means "unknown", not
    "the movie really costs $0". We flag them here; we fix them in `clean()`.
    """
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
    """
    Turn the "0 means unknown" values into real missing values (NaN).
    This matters a lot for later steps: a model must not learn that
    "budget = 0" is a normal value, and a mean/median would be biased
    if we kept fake zeros inside it.
    """
    df = df.copy()
    df["budget"] = df["budget"].replace(0, np.nan)
    df["revenue"] = df["revenue"].replace(0, np.nan)
    df["runtime"] = df["runtime"].replace(0, np.nan)

    # A rating must be between 0 and 10 -> anything else is an error, not a real value
    df.loc[(df["vote_average"] < 0) | (df["vote_average"] > 10), "vote_average"] = np.nan

    # A movie must have a title -> without one, the row is useless, we drop it
    before = len(df)
    df = df[df["title"].notna() & (df["title"].str.strip() != "")]
    removed = before - len(df)
    if removed:
        logger.info("Removed %s rows without a title", removed)

    return df


# --------------------------------------------------------------------------
# 4. Date conversion
# --------------------------------------------------------------------------
def convert_dates(df):
    """
    Convert release_date from text (e.g. "2010-05-14") to a real datetime.
    errors="coerce" means: if a date is broken or empty, put NaT (missing date)
    instead of crashing the whole program.
    """
    df = df.copy()
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    return df


# --------------------------------------------------------------------------
# 5. Split numeric / categorical / text / list columns
# --------------------------------------------------------------------------
def split_columns(df):
    """
    Return a dict grouping column names by type of data.
    We will reuse these lists later:
      - numeric    -> scaled and fed directly to models
      - categorical-> one-hot encoded
      - text       -> used with TF-IDF (Step 5)
      - list       -> counted / exploded (genres, keywords)
      - date       -> used to build year / month / decade features
    """
    groups = {
        "numeric": [c for c in NUMERIC_COLS if c in df.columns],
        "categorical": [c for c in CATEGORICAL_COLS if c in df.columns],
        "text": [c for c in TEXT_COLS if c in df.columns],
        "list": [c for c in LIST_COLS if c in df.columns],
        "date": [c for c in DATE_COLS if c in df.columns],
    }
    return groups


# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------
def clean(df):
    """Run every cleaning step in the right order and return a clean DataFrame."""
    df = drop_duplicates(df)
    df = fix_inconsistencies(df)
    df = convert_dates(df)

    # A missing overview becomes an empty string: TF-IDF (Step 5) can handle
    # "" perfectly, but it cannot handle NaN.
    df["overview"] = df["overview"].fillna("")

    return df


def run_cleaning():
    """Load, analyze, clean and save. This is what Airflow / the pipeline calls."""
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

    df_clean.to_pickle(DATA_PROCESSED / "movies_clean.pkl")  # pickle keeps list columns
    logger.info("Cleaned dataset saved: %s -> %s rows, %s columns",
                DATA_PROCESSED / "movies_clean.pkl", *df_clean.shape)
    return df_clean


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_cleaning()
