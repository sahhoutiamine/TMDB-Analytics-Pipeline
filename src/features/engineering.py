"""Step 4: feature engineering (no data leakage: nothing derived from the target source)."""
import pandas as pd

from src.config import DATA_PROCESSED


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["release_date"].dt.year
    df["month"] = df["release_date"].dt.month
    df["decade"] = (df["year"] // 10) * 10
    df["n_genres"] = df["genres"].apply(len)
    df["n_keywords"] = df["keywords"].apply(len)
    df["runtime_cat"] = pd.cut(df["runtime"], bins=[0, 90, 120, 150, 1000],
                               labels=["short", "medium", "long", "very_long"])
    df["overview_len"] = df["overview"].fillna("").str.split().str.len()
    # TODO: has_budget, roi (careful: revenue), is_english, ...
    return df


def run_features() -> pd.DataFrame:
    df = add_features(pd.read_pickle(DATA_PROCESSED / "movies_clean.pkl"))
    df.to_pickle(DATA_PROCESSED / "movies_features.pkl")
    return df
