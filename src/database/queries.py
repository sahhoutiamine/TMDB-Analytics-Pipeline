from src.database.mongo import get_collection


def top_rated_movies(min_votes=1000, limit=10):
    collection = get_collection()
    cursor = (
        collection.find(
            {"vote_count": {"$gte": min_votes}},
            {"_id": 0, "title": 1, "vote_average": 1, "vote_count": 1},
        )
        .sort("vote_average", -1)
        .limit(limit)
    )
    return list(cursor)


def movies_by_language(language_code, limit=10):
    collection = get_collection()
    cursor = collection.find(
        {"original_language": language_code},
        {"_id": 0, "title": 1, "original_language": 1, "vote_average": 1},
    ).limit(limit)
    return list(cursor)


def movies_released_after(year, limit=10):
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
    collection = get_collection()
    cursor = collection.find(
        {"keywords": keyword},
        {"_id": 0, "title": 1, "keywords": 1},
    ).limit(limit)
    return list(cursor)


def high_budget_movies(min_budget=100_000_000, limit=10):
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


def avg_rating_by_genre():
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
    for row in results:
        row["avg_rating"] = round(row["avg_rating"], 2)
    return results


def movie_count_by_decade():
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
