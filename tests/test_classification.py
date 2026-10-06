import numpy as np
import pandas as pd
import pytest

from src.models import classification


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


def test_build_target_uses_quantile_threshold(sample_df):
    df, threshold = classification.build_target(sample_df, quantile=0.75)
    assert threshold == sample_df["vote_count"].quantile(0.75)
    assert set(df["high_engagement"].unique()) <= {0, 1}
    assert df["high_engagement"].mean() == pytest.approx(0.25, abs=0.05)


def test_build_target_label_matches_threshold(sample_df):
    df, threshold = classification.build_target(sample_df, quantile=0.75)
    above = df[df["vote_count"] >= threshold]
    below = df[df["vote_count"] < threshold]
    assert (above["high_engagement"] == 1).all()
    assert (below["high_engagement"] == 0).all()


def test_prepare_features_excludes_vote_count_and_keyword_count(sample_df):
    df, _ = classification.build_target(sample_df)
    X, y = classification.prepare_features(df)
    assert "vote_count" not in X.columns
    assert "n_keywords" not in X.columns
    assert "vote_average" in X.columns
    assert "keywords_text" in X.columns
    assert y.name == "high_engagement"


def test_prepare_features_joins_keywords_into_text(sample_df):
    df, _ = classification.build_target(sample_df)
    X, _ = classification.prepare_features(df)
    assert X.loc[0, "keywords_text"] == "love war"
    assert X.loc[1, "keywords_text"] == "comedy"


def test_build_pipelines_returns_three_independent_models():
    pipelines = classification.build_pipelines()
    assert set(pipelines) == {"logistic_regression", "random_forest", "linear_svm"}
    preprocessors = [p.named_steps["preprocess"] for p in pipelines.values()]
    assert preprocessors[0] is not preprocessors[1]
    assert preprocessors[1] is not preprocessors[2]


def test_get_scores_logistic_regression_uses_predict_proba(sample_df):
    df, _ = classification.build_target(sample_df)
    X, y = classification.prepare_features(df)
    pipeline = classification.build_pipelines()["logistic_regression"]
    pipeline.fit(X, y)
    scores = classification.get_scores(pipeline, X)
    assert len(scores) == len(X)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_get_scores_linear_svm_uses_decision_function(sample_df):
    df, _ = classification.build_target(sample_df)
    X, y = classification.prepare_features(df)
    pipeline = classification.build_pipelines()["linear_svm"]
    pipeline.fit(X, y)
    scores = classification.get_scores(pipeline, X)
    assert len(scores) == len(X)


def test_evaluate_model_returns_expected_metrics(sample_df, capsys):
    df, _ = classification.build_target(sample_df)
    X, y = classification.prepare_features(df)
    pipeline = classification.build_pipelines()["logistic_regression"]
    pipeline.fit(X, y)
    metrics, cm = classification.evaluate_model("logistic_regression", pipeline, X, y)

    expected_keys = {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert set(metrics) == expected_keys
    for value in metrics.values():
        assert 0.0 <= value <= 1.0
    assert cm.shape == (2, 2)
    assert "logistic_regression" in capsys.readouterr().out


def test_run_classification_trains_and_saves_all_models(monkeypatch, tmp_path, sample_df, capsys):
    monkeypatch.setattr(classification, "load_features", lambda: sample_df)
    monkeypatch.setattr(classification, "MODELS_DIR", tmp_path)
    results, pipelines = classification.run_classification(test_size=0.3)

    assert set(results) == {"logistic_regression", "random_forest", "linear_svm"}
    for name in results:
        assert (tmp_path / f"classification_{name}.joblib").exists()

    out = capsys.readouterr().out
    assert "data leakage" in out
    assert "Best model by F1-score" in out