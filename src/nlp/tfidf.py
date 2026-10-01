import logging
import re

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import DATA_PROCESSED, MODELS_DIR

logger = logging.getLogger(__name__)

CONFIGS = [
    {"max_features": 1000, "ngram_range": (1, 1)},
    {"max_features": 5000, "ngram_range": (1, 1)},
    {"max_features": 5000, "ngram_range": (1, 2)},
    {"max_features": 10000, "ngram_range": (1, 2)},
]

FINAL_CONFIG = {"max_features": 5000, "ngram_range": (1, 2)}


def load_features():
    return pd.read_pickle(DATA_PROCESSED / "movies_features.pkl")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_overviews(df):
    overviews = df["overview"].fillna("")
    return overviews.apply(clean_text)


def build_tfidf(texts, max_features=5000, ngram_range=(1, 2)):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def top_terms(vectorizer, matrix, n=15):
    scores = np.asarray(matrix.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    order = np.argsort(scores)[::-1][:n]
    return [(terms[i], round(float(scores[i]), 2)) for i in order]


def run_experiments(texts):
    results = []
    for config in CONFIGS:
        vectorizer, matrix = build_tfidf(texts, **config)
        density = matrix.nnz / (matrix.shape[0] * matrix.shape[1]) * 100
        top = top_terms(vectorizer, matrix, n=10)

        print(f"-> max_features={config['max_features']}, "
              f"ngram_range={config['ngram_range']}: "
              f"matrix shape {matrix.shape}, density {density:.2f}%")
        print(f"   top terms: {[t for t, _ in top]}")

        results.append({"config": config, "shape": matrix.shape,
                        "density": density, "top_terms": top})
    return results


def run_tfidf():
    df = load_features()
    texts = prepare_overviews(df)

    empty_count = (texts == "").sum()
    print(f"-> {empty_count} movies have an empty overview after cleaning "
          f"({empty_count / len(texts) * 100:.1f}%). TfidfVectorizer handles "
          f"empty strings fine: they just produce an all-zero row.")

    run_experiments(texts)

    print(f"-> Final choice: max_features={FINAL_CONFIG['max_features']}, "
          f"ngram_range={FINAL_CONFIG['ngram_range']}. Bigrams capture short "
          f"phrases (e.g. 'based on') that single words lose, while "
          f"max_features=5000 keeps the matrix small enough to work with "
          f"quickly in clustering and recommendation.")

    vectorizer, matrix = build_tfidf(texts, **FINAL_CONFIG)
    print(f"-> Final matrix: {matrix.shape[0]} movies x {matrix.shape[1]} terms")
    print(f"-> Final top terms: {top_terms(vectorizer, matrix, n=15)}")

    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib")
    sparse.save_npz(DATA_PROCESSED / "tfidf_matrix.npz", matrix)
    np.save(DATA_PROCESSED / "tfidf_movie_ids.npy", df["movie_id"].values)

    logger.info("Saved vectorizer to %s", MODELS_DIR / "tfidf_vectorizer.joblib")
    logger.info("Saved matrix to %s", DATA_PROCESSED / "tfidf_matrix.npz")
    return vectorizer, matrix


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_tfidf()