import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def build_tfidf(texts: pd.Series, max_features: int = 5000, ngram_range=(1, 2)):
    vec = TfidfVectorizer(stop_words="english", max_features=max_features,
                          ngram_range=ngram_range, min_df=2)
    X = vec.fit_transform(texts.fillna("").map(clean_text))
    return vec, X

