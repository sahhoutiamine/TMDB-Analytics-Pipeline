"""
These tests use `mongomock`, a fake MongoDB that lives only in memory.
This lets us test our MongoDB code without needing Docker or a real
database running - and the tests run in a fraction of a second.
"""
import datetime

import mongomock
import pandas as pd
import pytest

from src.database import mongo, queries


@pytest.fixture
def fake_collection(monkeypatch):
    """Replace the real MongoClient with a fake one for every test in this file."""
    client = mongomock.MongoClient()
    monkeypatch.setattr(mongo, "get_client", lambda: client)
    return client["test_db"]["test_movies"]


@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {"movie_id": 1, "title": "Old Drama", "genres": ["Drama"],
         "vote_average": 7.0, "vote_count": 200,
         "release_date": pd.Timestamp("1995-06-01"), "budget": 1_000_000,
         "keywords": ["love"], "original_language": "en"},
        {"movie_id": 2, "title": "New Comedy", "genres": ["Comedy", "Drama"],
         "vote_average": 5.0, "vote_count": 80,
         "release_date": pd.Timestamp("2015-06-01"), "budget": 50_000_000,
         "keywords": ["friendship"], "original_language": "fr"},
        {"movie_id": 3, "title": "Unrated", "genres": ["Action"],
         "vote_average": None, "vote_count": 2,   # too few votes -> excluded from aggregation
         "release_date": pd.NaT, "budget": None,
         "keywords": [], "original_language": "en"},
    ])


# --------------------------------------------------------------------------
# mongo.py
# --------------------------------------------------------------------------
def test_load_to_mongo_inserts_all_rows(monkeypatch, fake_collection, sample_df):
    monkeypatch.setattr(mongo, "get_collection", lambda: fake_collection)
    n_inserted = mongo.load_to_mongo(sample_df)
    assert n_inserted == 3
    assert fake_collection.count_documents({}) == 3


def test_load_to_mongo_replaces_old_data(monkeypatch, fake_collection, sample_df):
    monkeypatch.setattr(mongo, "get_collection", lambda: fake_collection)
    fake_collection.insert_one({"movie_id": 999, "title": "Should disappear"})
    mongo.load_to_mongo(sample_df)
    # the old fake document must be gone: load_to_mongo always starts empty
    assert fake_collection.find_one({"movie_id": 999}) is None
    assert fake_collection.count_documents({}) == 3


def test_load_to_mongo_converts_nan_to_none(monkeypatch, fake_collection, sample_df):
    monkeypatch.setattr(mongo, "get_collection", lambda: fake_collection)
    mongo.load_to_mongo(sample_df)
    doc = fake_collection.find_one({"movie_id": 3})
    assert doc["vote_average"] is None  # NaN must become None, not stay NaN
    assert doc["budget"] is None


def test_dataframe_to_documents_length_matches_rows(sample_df):
    docs = mongo.dataframe_to_documents(sample_df)
    assert len(docs) == len(sample_df)


# --------------------------------------------------------------------------
# queries.py
# --------------------------------------------------------------------------
@pytest.fixture
def loaded_collection(monkeypatch, fake_collection, sample_df):
    """A fake collection already filled with the sample data."""
    monkeypatch.setattr(queries, "get_collection", lambda: fake_collection)
    fake_collection.insert_many(mongo.dataframe_to_documents(sample_df))
    return fake_collection


def test_top_rated_movies_orders_by_rating(loaded_collection):
    results = queries.top_rated_movies(min_votes=50, limit=10)
    titles = [r["title"] for r in results]
    assert titles == ["Old Drama", "New Comedy"]  # 7.0 before 5.0; "Unrated" excluded (2 votes)


def test_movies_by_language_filters_correctly(loaded_collection):
    results = queries.movies_by_language("fr")
    assert len(results) == 1
    assert results[0]["title"] == "New Comedy"


def test_movies_released_after_filters_by_year(loaded_collection):
    results = queries.movies_released_after(2000)
    titles = [r["title"] for r in results]
    assert titles == ["New Comedy"]  # only the 2015 movie qualifies


def test_search_by_keyword_finds_matching_movies(loaded_collection):
    results = queries.search_by_keyword("love")
    assert len(results) == 1
    assert results[0]["title"] == "Old Drama"


def test_high_budget_movies_filters_and_sorts(loaded_collection):
    results = queries.high_budget_movies(min_budget=10_000_000)
    assert [r["title"] for r in results] == ["New Comedy"]


def test_avg_rating_by_genre_aggregation(loaded_collection):
    results = queries.avg_rating_by_genre()
    by_genre = {r["genre"]: r for r in results}
    # "Drama" appears in both rated movies (7.0 and 5.0) -> average 6.0
    assert by_genre["Drama"]["avg_rating"] == 6.0
    assert by_genre["Drama"]["n_movies"] == 2
    # "Action" belongs only to the movie with 2 votes -> filtered out by $match
    assert "Action" not in by_genre


def test_movie_count_by_decade_aggregation(loaded_collection):
    results = queries.movie_count_by_decade()
    by_decade = {r["decade"]: r["n_movies"] for r in results}
    assert by_decade[1990] == 1
    assert by_decade[2010] == 1
    # the movie with a missing (NaT) release_date must not appear in any decade
    assert sum(by_decade.values()) == 2
