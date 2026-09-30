import numpy as np
import pandas as pd
import pytest

from src.cleaning import clean


def make_raw_df():
    return pd.DataFrame([
        {"movie_id": 1, "title": "Movie A", "overview": "A story", "release_date": "2010-05-01",
         "runtime": 100, "original_language": "en", "genres": ["Drama"], "keywords": ["love"],
         "budget": 1_000_000, "revenue": 5_000_000, "popularity": 10.0, "vote_average": 7.5, "vote_count": 200},
        {"movie_id": 2, "title": "Movie B", "overview": None, "release_date": "2015-01-10",
         "runtime": 0, "original_language": "fr", "genres": ["Comedy"], "keywords": [],
         "budget": 0, "revenue": 0, "popularity": 3.0, "vote_average": 6.0, "vote_count": 50},
        {"movie_id": 2, "title": "Movie B", "overview": None, "release_date": "2015-01-10",
         "runtime": 0, "original_language": "fr", "genres": ["Comedy"], "keywords": [],
         "budget": 0, "revenue": 0, "popularity": 3.0, "vote_average": 6.0, "vote_count": 50},
        {"movie_id": 3, "title": "", "overview": "No title here", "release_date": "not-a-date",
         "runtime": 90, "original_language": "en", "genres": [], "keywords": [],
         "budget": 500, "revenue": 500, "popularity": 1.0, "vote_average": 15.0, "vote_count": 1},
    ])


def test_read_one_movie_extracts_expected_fields():
    raw = {"id": 1, "title": "X", "genres": [{"name": "Drama"}],
           "keywords": {"keywords": [{"name": "love"}]}}
    row = clean.read_one_movie(raw)
    assert row["movie_id"] == 1
    assert row["genres"] == ["Drama"]
    assert row["keywords"] == ["love"]


def test_read_one_movie_handles_missing_fields():
    row = clean.read_one_movie({"id": 1, "title": "X"})
    assert row["genres"] == [] and row["keywords"] == []


def test_report_duplicates_counts_movie_id():
    df = make_raw_df()
    report = clean.report_duplicates(df)
    assert report["duplicate_movie_id"] == 1


def test_drop_duplicates_removes_repeated_id():
    df = make_raw_df()
    out = clean.drop_duplicates(df)
    assert out["movie_id"].duplicated().sum() == 0
    assert len(out) == 3


def test_detect_inconsistencies_finds_the_planted_problems():
    df = make_raw_df()
    report = clean.detect_inconsistencies(df)
    assert report["budget_zero"] == 2
    assert report["runtime_zero_or_null"] == 2
    assert report["overview_empty"] == 2
    assert report["vote_average_out_of_range"] == 1
    assert report["release_date_missing_or_invalid"] == 1


def test_fix_inconsistencies_turns_zero_into_nan():
    df = make_raw_df()
    out = clean.fix_inconsistencies(df)
    zero_budget_rows = out[out["movie_id"] == 2]
    assert zero_budget_rows["budget"].isna().all()
    assert zero_budget_rows["runtime"].isna().all()


def test_fix_inconsistencies_removes_bad_rating_range():
    df = make_raw_df()
    out = clean.fix_inconsistencies(df)
    row3 = out[out["movie_id"] == 3]
    assert row3.empty or row3["vote_average"].isna().all()


def test_fix_inconsistencies_drops_rows_without_title():
    df = make_raw_df()
    out = clean.fix_inconsistencies(df)
    assert 3 not in out["movie_id"].values


def test_convert_dates_creates_datetime_and_handles_bad_dates():
    df = make_raw_df()
    out = clean.convert_dates(df)
    assert pd.api.types.is_datetime64_any_dtype(out["release_date"])
    bad_row = out[out["movie_id"] == 3]
    assert bad_row["release_date"].isna().all()


def test_split_columns_groups_are_correct():
    df = make_raw_df()
    groups = clean.split_columns(df)
    assert "budget" in groups["numeric"]
    assert "original_language" in groups["categorical"]
    assert "overview" in groups["text"]
    assert "genres" in groups["list"]
    assert "release_date" in groups["date"]


def test_clean_end_to_end_runs_without_error():
    df = make_raw_df()
    out = clean.clean(df)
    assert out["movie_id"].duplicated().sum() == 0
    assert out["overview"].isna().sum() == 0
    assert pd.api.types.is_datetime64_any_dtype(out["release_date"])
