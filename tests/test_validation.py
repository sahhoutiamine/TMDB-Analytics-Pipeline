import numpy as np
import pandas as pd
import pytest

from src.models import classification, validation


@pytest.fixture
def sample_df():
    n = 100
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "movie_id": range(n),
        "runtime": rng.integers(70, 180, n).astype(float),
        "budget": rng.integers(1000, 100_000_000, n).astype(float),
        "revenue": rng.integers(1000, 500_000_000, n).astype(float),
        "popularity": rng.uniform(1, 500, n),
        "vote_average": rng.uniform(1, 10, n),
        "vote_count": rng.integers(1, 30000, n),
        "release_year": rng.integers(1990, 2024, n),
        "release_month": rng.integers(1, 12, n),
        "release_decade": rng.choice([1990, 2000, 2010, 2020], n),
        "n_genres": rng.integers(1, 4, n),
        "overview_word_count": rng.integers(5, 80, n),
        "budget_known": rng.integers(0, 2, n),
        "revenue_known": rng.integers(0, 2, n),
        "original_language": rng.choice(["en", "fr", "es"], n),
        "runtime_category": rng.choice(["short", "medium", "long"], n),
        "keywords": [["love", "war"] if i % 2 == 0 else ["comedy"] for i in range(n)],
    })


@pytest.fixture
def Xy(sample_df):
    df, _ = classification.build_target(sample_df)
    return classification.prepare_features(df)


def test_run_cross_validation_returns_all_metrics(Xy, capsys):
    X, y = Xy
    pipelines = {"random_forest": classification.build_pipelines()["random_forest"]}
    results = validation.run_cross_validation(X, y, pipelines, n_splits=3)

    assert set(results["random_forest"]) == set(validation.CV_SCORING)
    for value in results["random_forest"].values():
        assert 0.0 <= value <= 1.0
    assert "random_forest" in capsys.readouterr().out


def test_run_grid_search_finds_best_params(Xy):
    X, y = Xy
    pipeline = classification.build_pipelines()["random_forest"]
    small_grid = {"classifier__n_estimators": [50, 100]}
    search = validation.run_grid_search(X, y, pipeline, param_grid=small_grid, n_splits=3)

    assert search.best_params_["classifier__n_estimators"] in [50, 100]
    assert 0.0 <= search.best_score_ <= 1.0
    assert hasattr(search, "best_estimator_")


def test_run_validation_end_to_end(monkeypatch, tmp_path, sample_df, capsys):
    monkeypatch.setattr(validation, "load_features", lambda: sample_df)
    monkeypatch.setattr(validation, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(validation, "PARAM_GRID", {"classifier__n_estimators": [50]})

    before, search = validation.run_validation()

    assert set(before) == {"logistic_regression", "random_forest", "linear_svm"}
    assert (tmp_path / "classification_random_forest_tuned.joblib").exists()

    out = capsys.readouterr().out
    assert "Best parameters found" in out
    assert "before tuning" in out
    assert "after tuning" in out