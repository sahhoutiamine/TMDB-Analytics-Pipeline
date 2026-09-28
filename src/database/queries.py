"""Step 2: example MongoDB queries + aggregations."""
from src.database.mongo import get_collection


def top_rated(min_votes: int = 1000, limit: int = 10):
    return list(get_collection().find(
        {"vote_count": {"$gte": min_votes}}, {"_id": 0, "title": 1, "vote_average": 1}
    ).sort("vote_average", -1).limit(limit))


def avg_rating_by_genre():
    pipeline = [
        {"$unwind": "$genres"},
        {"$match": {"vote_count": {"$gte": 50}}},
        {"$group": {"_id": "$genres", "avg_rating": {"$avg": "$vote_average"}, "n": {"$sum": 1}}},
        {"$sort": {"avg_rating": -1}},
        {"$project": {"_id": 0, "genre": "$_id", "avg_rating": {"$round": ["$avg_rating", 2]}, "n": 1}},
    ]
    return list(get_collection().aggregate(pipeline))

# TODO: add more queries (by language, by decade, budget > X, text search...)
