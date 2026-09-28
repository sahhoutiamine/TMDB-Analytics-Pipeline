"""Step 9 (bonus): content-based recommender (TF-IDF + cosine similarity)."""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def recommend(title: str, df: pd.DataFrame, tfidf_matrix, top_n: int = 5) -> pd.DataFrame:
    idx = df.index[df["title"].str.lower() == title.lower()]
    if len(idx) == 0:
        raise ValueError(f"Movie not found: {title}")
    i = df.index.get_loc(idx[0])
    sims = cosine_similarity(tfidf_matrix[i], tfidf_matrix).ravel()
    sims[i] = -1  # exclude the movie itself
    top = np.argsort(sims)[::-1][:top_n]
    out = df.iloc[top][["title", "genres", "year"]].copy()
    out["similarity"] = sims[top].round(3)
    return out
