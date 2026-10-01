import pandas as pd
import pytest

from src.features import engineering


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "movie_id": [1, 2, 3],
        "title": ["A", "B", "C"],
        "overview": ["a short plot", "", "a much longer movie plot summary here"],
        "release_date": pd.to_datetime(["2015-06-15", "2005-01-01", None]),
        "runtime": [85.0, 130.0, None],
        "original_language": ["en", "fr", "en"],
        "genres": [["Drama"], ["Comedy", "Action"], []],
        "keywords": [["love"], [], ["war", "hero"]],
        "budget": [1_000_000, None, 500_000],
        "revenue": [None, None, 2_000_000],
        "popularity": [10.0, 5.0, 2.0],
        "vote_average": [7.0, 6.0, 8.0],
        "vote_count": [100, 50, 10],
    })


def test_add_date_parts(sample_df):
    out = engineering.add_date_parts(sample_df)
    assert out.loc[0, "release_year"] == 2015
    assert out.loc[0, "release_month"] == 6
    assert out.loc[0, "release_decade"] == 2010
    assert out.loc[1, "release_decade"] == 2000
    assert pd.isna(out.loc[2, "release_year"])


def test_add_genre_keyword_counts(sample_df):
    out = engineering.add_genre_keyword_counts(sample_df)
    assert out.loc[0, "n_genres"] == 1
    assert out.loc[1, "n_genres"] == 2
    assert out.loc[2, "n_genres"] == 0
    assert out.loc[2, "n_keywords"] == 2


def test_add_runtime_category(sample_df):
    out = engineering.add_runtime_category(sample_df)
    assert out.loc[0, "runtime_category"] == "short"
    assert out.loc[1, "runtime_category"] == "long"
    assert pd.isna(out.loc[2, "runtime_category"])


def test_add_overview_length(sample_df):
    out = engineering.add_overview_length(sample_df)
    assert out.loc[0, "overview_word_count"] == 3
    assert out.loc[1, "overview_word_count"] == 0
    assert out.loc[2, "overview_word_count"] == 7


def test_add_language_flag(sample_df):
    out = engineering.add_language_flag(sample_df)
    assert out.loc[0, "is_english"] == 1
    assert out.loc[1, "is_english"] == 0


def test_add_missingness_flags(sample_df):
    out = engineering.add_missingness_flags(sample_df)
    assert out.loc[0, "budget_known"] == 1
    assert out.loc[1, "budget_known"] == 0
    assert out.loc[0, "revenue_known"] == 0
    assert out.loc[2, "revenue_known"] == 1


def test_add_features_creates_all_columns(sample_df):
    out = engineering.add_features(sample_df)
    expected = {"release_year", "release_month", "release_decade", "n_genres",
                "n_keywords", "runtime_category", "overview_word_count",
                "is_english", "budget_known", "revenue_known"}
    assert expected.issubset(out.columns)


def test_add_features_does_not_use_vote_count(sample_df):
    import inspect
    source = inspect.getsource(engineering.add_features)
    for fn in [engineering.add_date_parts, engineering.add_genre_keyword_counts,
               engineering.add_runtime_category, engineering.add_overview_length,
               engineering.add_language_flag, engineering.add_missingness_flags]:
        assert "vote_count" not in inspect.getsource(fn)
    assert "vote_count" not in source


def test_run_features_saves_pickle(monkeypatch, tmp_path, sample_df, capsys):
    monkeypatch.setattr(engineering, "load_clean", lambda: sample_df)
    monkeypatch.setattr(engineering, "DATA_PROCESSED", tmp_path)
    out = engineering.run_features()
    assert (tmp_path / "movies_features.pkl").exists()
    assert "vote_count is NOT used" in capsys.readouterr().out
    assert len(out) == 3