import logging

import pandas as pd

from src.config import DATA_PROCESSED

logger = logging.getLogger(__name__)

RUNTIME_BINS = [0, 90, 120, 150, 10_000]
RUNTIME_LABELS = ["short", "medium", "long", "very_long"]


def load_clean():
    return pd.read_pickle(DATA_PROCESSED / "movies_clean.pkl")


def add_date_parts(df):
    df = df.copy()
    df["release_year"] = df["release_date"].dt.year
    df["release_month"] = df["release_date"].dt.month
    df["release_decade"] = (df["release_year"] // 10) * 10
    return df


def add_genre_keyword_counts(df):
    df = df.copy()
    df["n_genres"] = df["genres"].apply(len)
    df["n_keywords"] = df["keywords"].apply(len)
    return df


def add_runtime_category(df):
    df = df.copy()
    df["runtime_category"] = pd.cut(df["runtime"], bins=RUNTIME_BINS, labels=RUNTIME_LABELS)
    return df


def add_overview_length(df):
    df = df.copy()
    df["overview_word_count"] = df["overview"].fillna("").str.split().str.len()
    return df


def add_language_flag(df):
    df = df.copy()
    df["is_english"] = (df["original_language"] == "en").astype(int)
    return df


def add_missingness_flags(df):
    df = df.copy()
    df["budget_known"] = df["budget"].notna().astype(int)
    df["revenue_known"] = df["revenue"].notna().astype(int)
    return df


def add_features(df):
    df = add_date_parts(df)
    df = add_genre_keyword_counts(df)
    df = add_runtime_category(df)
    df = add_overview_length(df)
    df = add_language_flag(df)
    df = add_missingness_flags(df)
    return df


def print_justifications(df):
    print(f"-> release_year / release_month / release_decade: captures trends over "
          f"time (e.g. more releases recently). Range: {int(df['release_year'].min())} "
          f"to {int(df['release_year'].max())}.")

    print(f"-> n_genres / n_keywords: movies with more tags may reach a wider "
          f"audience. Average: {df['n_genres'].mean():.1f} genres, "
          f"{df['n_keywords'].mean():.1f} keywords per movie.")

    print("-> runtime_category: groups runtime into simple buckets, easier for "
          "a model to use than a raw number spread across a wide range.")
    print(df["runtime_category"].value_counts())

    print(f"-> overview_word_count: a longer synopsis may signal a more "
          f"detailed, higher-budget production. Median: "
          f"{df['overview_word_count'].median():.0f} words.")

    print(f"-> is_english: original language is a simple, known-in-advance "
          f"signal. {df['is_english'].mean() * 100:.0f}% of movies are in English.")

    print(f"-> budget_known / revenue_known: missingness itself can be "
          f"informative (e.g. smaller/independent movies report less data). "
          f"{df['budget_known'].mean() * 100:.0f}% have a known budget.")

    print("-> vote_count is NOT used here: it will build the classification "
          "target in Step 6, so using it (or anything derived from it) as a "
          "feature would be data leakage.")


def run_features():
    df = load_clean()
    df = add_features(df)
    print_justifications(df)
    df.to_pickle(DATA_PROCESSED / "movies_features.pkl")
    logger.info("Features saved: %s -> %s rows, %s columns",
                DATA_PROCESSED / "movies_features.pkl", *df.shape)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_features()