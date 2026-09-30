"""
Quick check of the downloaded data.
Run it with:   python -m src.extraction.inspect_raw

(The deeper analysis - types, missing values, duplicates, inconsistencies -
lives in src/cleaning/clean.py, which is Step 2. This script is just a fast
"did the extraction work?" check for Step 1.)
"""
import json

import pandas as pd

from src.cleaning.clean import read_one_movie
from src.config import RAW_MOVIES_FILE


def main():
    if not RAW_MOVIES_FILE.exists():
        print("Nothing to inspect. Run: python -m src.extraction.extract")
        return

    with open(RAW_MOVIES_FILE, "r", encoding="utf-8") as f:
        movie_list = json.load(f)
    print(f"Movies in {RAW_MOVIES_FILE.name}: {len(movie_list)}")

    df = pd.DataFrame([read_one_movie(m) for m in movie_list])
    print("\nShape (rows, columns):", df.shape)
    print("\nFirst 5 rows:")
    print(df[["movie_id", "title", "release_date", "vote_average", "vote_count"]].head())
    print("\nMissing values per column:")
    print(df.isna().sum())
    print("\nOverview empty:", (df["overview"].fillna("") == "").sum())
    print("Budget = 0:", (df["budget"] == 0).sum(), "| Revenue = 0:", (df["revenue"] == 0).sum())


if __name__ == "__main__":
    main()
