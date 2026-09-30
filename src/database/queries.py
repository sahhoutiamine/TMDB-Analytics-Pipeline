"""
MongoDB queries  (STEP 2)
=========================
Run it with:   python -m src.database.queries

A "query" asks MongoDB for documents that match a condition.
An "aggregation" is a pipeline: several steps chained together, each one
transforming the data a bit more (filter -> group -> sort -> reshape).

Every function here returns a plain Python list, so it's easy to print,
test, or show in Streamlit later.
"""
from src.database.mongo import get_collection


# --------------------------------------------------------------------------
# Simple queries (find)
# --------------------------------------------------------------------------
def top_rated_movies(min_votes=1000, limit=10):
    """The best-rated movies, but only among movies enough people voted for
    (a movie with 5 votes at 10/10 is not reliably "the best")."""
    collection = get_collection()
    cursor = (
        collection.find(
            {"vote_count": {"$gte": min_votes}},          # filter
            {"_id": 0, "title": 1, "vote_average": 1, "vote_count": 1},  # only these fields
        )
        .sort("vote_average", -1)  # -1 = descending (highest first)
        .limit(limit)
    )
    return list(cursor)


def movies_by_language(language_code, limit=10):
    """Movies in a given original language, e.g. "fr" for French, "en" for English."""
    collection = get_collection()
    cursor = collection.find(
        {"original_language": language_code},
        {"_id": 0, "title": 1, "original_language": 1, "vote_average": 1},
    ).limit(limit)
    return list(cursor)


def movies_released_after(year, limit=10):
    """Movies released on or after a given year, most popular first."""
    import datetime
    collection = get_collection()
    cursor = (
        collection.find(
            {"release_date": {"$gte": datetime.datetime(year, 1, 1)}},
            {"_id": 0, "title": 1, "release_date": 1, "popularity": 1},
        )
        .sort("popularity", -1)
        .limit(limit)
    )
    return list(cursor)


def search_by_keyword(keyword, limit=10):
    """
    Movies that contain a given keyword.
    `keywords` is stored as an array in MongoDB, so this simple filter
    automatically checks "is `keyword` one of the items in the array?".
    """
    collection = get_collection()
    cursor = collection.find(
        {"keywords": keyword},
        {"_id": 0, "title": 1, "keywords": 1},
    ).limit(limit)
    return list(cursor)


def high_budget_movies(min_budget=100_000_000, limit=10):
    """Movies with a budget of at least `min_budget`, biggest budget first."""
    collection = get_collection()
    cursor = (
        collection.find(
            {"budget": {"$gte": min_budget}},
            {"_id": 0, "title": 1, "budget": 1, "revenue": 1},
        )
        .sort("budget", -1)
        .limit(limit)
    )
    return list(cursor)


# --------------------------------------------------------------------------
# Aggregations (pipelines)
# --------------------------------------------------------------------------
def avg_rating_by_genre():
    """
    Average rating per genre.

    Pipeline steps:
      $unwind  -> a movie with genres ["Action", "Drama"] becomes 2 rows,
                  one per genre (needed because genres is a list)
      $match   -> keep only movies with at least 50 votes (reliable ratings)
      $group   -> one group per genre, compute the average rating and count
      $sort    -> best average rating first
      $project -> keep only the fields we want (genre, avg_rating, n_movies)
    """
    pipeline = [
        {"$unwind": "$genres"},
        {"$match": {"vote_count": {"$gte": 50}}},
        {"$group": {
            "_id": "$genres",
            "avg_rating": {"$avg": "$vote_average"},
            "n_movies": {"$sum": 1},
        }},
        {"$sort": {"avg_rating": -1}},
        {"$project": {
            "_id": 0,
            "genre": "$_id",
            "avg_rating": 1,
            "n_movies": 1,
        }},
    ]
    results = list(get_collection().aggregate(pipeline))
    # Round in Python: simpler to read than $round, and works the same everywhere
    for row in results:
        row["avg_rating"] = round(row["avg_rating"], 2)
    return results


def movie_count_by_decade():
    """
    Number of movies released per decade.

    Pipeline steps:
      $match   -> ignore movies without a release date
      $project -> compute the decade from release_date ( e.g. 2015 -> 2010 )
      $group   -> count how many movies fall in each decade
      $sort    -> oldest decade first
    """
    pipeline = [
        {"$match": {"release_date": {"$ne": None}}},
        {"$project": {
            "decade": {
                "$multiply": [{"$floor": {"$divide": [{"$year": "$release_date"}, 10]}}, 10]
            }
        }},
        {"$group": {"_id": "$decade", "n_movies": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
        {"$project": {"_id": 0, "decade": "$_id", "n_movies": 1}},
    ]
    return list(get_collection().aggregate(pipeline))


if __name__ == "__main__":
    print("Top rated movies:")
    for m in top_rated_movies(limit=5):
        print(" ", m)

    print("\nAverage rating by genre:")
    for g in avg_rating_by_genre():
        print(" ", g)

    print("\nMovies per decade:")
    for d in movie_count_by_decade():
        print(" ", d)
