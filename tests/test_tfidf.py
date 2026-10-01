import numpy as np
import pandas as pd
import pytest

from src.nlp import tfidf


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "movie_id": [1, 2, 3, 4],
        "overview": [
            "A SPACE Hero fights aliens in space!!",
            "a space hero fights aliens in space again",
            None,
            "A romantic comedy about love and friendship",
        ],
    })


def test_clean_text_lowercases_and_removes_punctuation():
    out = tfidf.clean_text("Hello, World! 123")
    assert out == "hello world"


def test_clean_text_collapses_whitespace():
    out = tfidf.clean_text("too    many   spaces")
    assert out == "too many spaces"


def test_clean_text_handles_none():
    assert tfidf.clean_text(None) == "none"


def test_prepare_overviews_fills_missing(sample_df):
    out = tfidf.prepare_overviews(sample_df)
    assert out.isna().sum() == 0
    assert out.iloc[2] == ""


def test_prepare_overviews_cleans_text(sample_df):
    out = tfidf.prepare_overviews(sample_df)
    assert out.iloc[0] == "a space hero fights aliens in space"


def test_build_tfidf_returns_correct_shape(sample_df):
    texts = tfidf.prepare_overviews(sample_df)
    vectorizer, matrix = tfidf.build_tfidf(texts, max_features=50, ngram_range=(1, 1))
    assert matrix.shape[0] == len(sample_df)
    assert matrix.shape[1] <= 50


def test_build_tfidf_respects_ngram_range(sample_df):
    texts = tfidf.prepare_overviews(sample_df)
    _, matrix_unigram = tfidf.build_tfidf(texts, max_features=50, ngram_range=(1, 1))
    _, matrix_bigram = tfidf.build_tfidf(texts, max_features=50, ngram_range=(1, 2))
    assert matrix_bigram.shape[1] >= matrix_unigram.shape[1]


def test_top_terms_returns_sorted_list(sample_df):
    texts = tfidf.prepare_overviews(sample_df)
    vectorizer, matrix = tfidf.build_tfidf(texts, max_features=50, ngram_range=(1, 1))
    top = tfidf.top_terms(vectorizer, matrix, n=5)
    scores = [score for _, score in top]
    assert scores == sorted(scores, reverse=True)
    assert len(top) <= 5


def test_run_experiments_covers_all_configs(sample_df, capsys):
    texts = tfidf.prepare_overviews(sample_df)
    results = tfidf.run_experiments(texts)
    assert len(results) == len(tfidf.CONFIGS)
    for r in results:
        assert r["shape"][0] == len(sample_df)
    assert "max_features" in capsys.readouterr().out


def test_run_tfidf_saves_files(monkeypatch, tmp_path, sample_df, capsys):
    monkeypatch.setattr(tfidf, "load_features", lambda: sample_df)
    monkeypatch.setattr(tfidf, "DATA_PROCESSED", tmp_path)
    monkeypatch.setattr(tfidf, "MODELS_DIR", tmp_path)
    vectorizer, matrix = tfidf.run_tfidf()

    assert (tmp_path / "tfidf_vectorizer.joblib").exists()
    assert (tmp_path / "tfidf_matrix.npz").exists()
    assert (tmp_path / "tfidf_movie_ids.npy").exists()

    saved_ids = np.load(tmp_path / "tfidf_movie_ids.npy")
    assert list(saved_ids) == [1, 2, 3, 4]
    assert matrix.shape[0] == 4