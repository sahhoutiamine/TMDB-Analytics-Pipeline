import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models import clustering


@pytest.fixture
def sample_df():
    n = 60
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "movie_id": range(n),
        "runtime": rng.integers(70, 180, n).astype(float),
        "budget": rng.integers(1000, 100_000_000, n).astype(float),
        "revenue": rng.integers(1000, 500_000_000, n).astype(float),
        "popularity": rng.uniform(1, 500, n),
        "vote_average": rng.uniform(1, 10, n),
        "n_genres": rng.integers(1, 4, n),
        "overview_word_count": rng.integers(5, 80, n),
        "release_year": rng.integers(1990, 2024, n),
        "genres": [["Drama", "Action"] if i % 2 == 0 else ["Comedy"] for i in range(n)],
    })


@pytest.fixture
def tfidf_matrix_and_vectorizer():
    texts = [
        "a space hero fights aliens", "aliens invade the space station",
        "a love story in paris", "two people fall in love in paris",
        "a war movie about soldiers", "soldiers fight a war in the trenches",
    ] * 10
    vectorizer = TfidfVectorizer(max_features=50)
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def test_build_structured_matrix_shape(sample_df):
    X = clustering.build_structured_matrix(sample_df)
    assert X.shape == (len(sample_df), len(clustering.STRUCTURED_FEATURES))


def test_build_structured_matrix_handles_missing(sample_df):
    sample_df.loc[0, "budget"] = np.nan
    X = clustering.build_structured_matrix(sample_df)
    assert not np.isnan(X).any()


def test_find_best_k_returns_valid_k(sample_df):
    X = clustering.build_structured_matrix(sample_df)
    best_k, scores = clustering.find_best_k(X, k_values=[2, 3, 4], label="test")
    assert best_k in [2, 3, 4]
    assert set(scores) == {2, 3, 4}
    for score in scores.values():
        assert -1.0 <= score <= 1.0


def test_interpret_structured_clusters_adds_cluster_column(sample_df):
    labels = np.array([0, 1] * (len(sample_df) // 2))
    out = clustering.interpret_structured_clusters(sample_df, labels)
    assert "cluster" in out.columns
    assert set(out["cluster"].unique()) == {0, 1}


def test_run_structured_clustering_saves_files(monkeypatch, tmp_path, sample_df):
    monkeypatch.setattr(clustering, "load_features", lambda: sample_df)
    monkeypatch.setattr(clustering, "DATA_PROCESSED", tmp_path)
    monkeypatch.setattr(clustering, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(clustering, "K_VALUES", [2, 3])

    df, kmeans, scores = clustering.run_structured_clustering()

    assert (tmp_path / "kmeans_structured.joblib").exists()
    assert (tmp_path / "movies_clustered_structured.pkl").exists()
    assert "cluster" in df.columns
    assert set(scores) == {2, 3}


def test_top_terms_per_cluster_returns_terms_per_cluster(tfidf_matrix_and_vectorizer):
    vectorizer, matrix = tfidf_matrix_and_vectorizer
    labels = np.array([0, 0, 1, 1, 2, 2] * 10)
    result = clustering.top_terms_per_cluster(vectorizer, matrix, labels, n_terms=5)
    assert set(result) == {0, 1, 2}
    for terms in result.values():
        assert len(terms) <= 5


def test_plot_tfidf_clusters_saves_file(monkeypatch, tmp_path, tfidf_matrix_and_vectorizer):
    monkeypatch.setattr(clustering, "FIGURES_DIR", tmp_path)
    vectorizer, matrix = tfidf_matrix_and_vectorizer
    labels = np.array([0, 0, 1, 1, 2, 2] * 10)
    clustering.plot_tfidf_clusters(matrix, labels)
    assert (tmp_path / "10_tfidf_clusters_2d.png").exists()


def test_run_tfidf_clustering_saves_files(monkeypatch, tmp_path, tfidf_matrix_and_vectorizer):
    vectorizer, matrix = tfidf_matrix_and_vectorizer
    movie_ids = np.arange(matrix.shape[0])

    monkeypatch.setattr(clustering, "load_tfidf", lambda: (matrix, movie_ids))
    monkeypatch.setattr(clustering, "DATA_PROCESSED", tmp_path)
    monkeypatch.setattr(clustering, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(clustering, "FIGURES_DIR", tmp_path)
    monkeypatch.setattr(clustering, "K_VALUES", [2, 3])
    joblib.dump(vectorizer, tmp_path / "tfidf_vectorizer.joblib")

    labels, kmeans, scores, top_terms = clustering.run_tfidf_clustering()

    assert (tmp_path / "kmeans_tfidf.joblib").exists()
    assert (tmp_path / "tfidf_cluster_labels.npy").exists()
    assert (tmp_path / "10_tfidf_clusters_2d.png").exists()
    assert len(labels) == matrix.shape[0]
    assert set(scores) == {2, 3}


def test_run_clustering_end_to_end(monkeypatch, tmp_path, sample_df, tfidf_matrix_and_vectorizer):
    vectorizer, matrix = tfidf_matrix_and_vectorizer
    movie_ids = np.arange(matrix.shape[0])

    monkeypatch.setattr(clustering, "load_features", lambda: sample_df)
    monkeypatch.setattr(clustering, "load_tfidf", lambda: (matrix, movie_ids))
    monkeypatch.setattr(clustering, "DATA_PROCESSED", tmp_path)
    monkeypatch.setattr(clustering, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(clustering, "FIGURES_DIR", tmp_path)
    monkeypatch.setattr(clustering, "K_VALUES", [2, 3])
    joblib.dump(vectorizer, tmp_path / "tfidf_vectorizer.joblib")

    results = clustering.run_clustering()

    assert "structured" in results
    assert "tfidf" in results