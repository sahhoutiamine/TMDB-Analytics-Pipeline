"""Extraction pipeline: paginate discover -> fetch details -> save raw JSON to data/raw/."""
import json
import logging

from src.config import DATA_RAW, N_MOVIES
from src.extraction.tmdb_client import TMDBClient

logger = logging.getLogger(__name__)
MAX_PAGES = 500  # TMDB limit


def run_extraction(n_movies: int = N_MOVIES, force: bool = False) -> int:
    """Download movies; skip those already cached in data/raw/ (avoid useless API calls)."""
    client = TMDBClient()
    ids: list[int] = []
    for page in range(1, MAX_PAGES + 1):
        if len(ids) >= n_movies:
            break
        results = client.discover_page(page, **{"vote_count.gte": 50})
        if not results:
            logger.info("Empty page %s, stopping pagination", page)
            break
        ids.extend(r["id"] for r in results)
    ids = list(dict.fromkeys(ids))[:n_movies]

    saved = 0
    for i, mid in enumerate(ids, 1):
        path = DATA_RAW / f"movie_{mid}.json"
        if path.exists() and not force:
            continue
        details = client.movie_details(mid)
        if not details:
            continue
        path.write_text(json.dumps(details, ensure_ascii=False), encoding="utf-8")
        saved += 1
        if i % 100 == 0:
            logger.info("%s/%s movies processed", i, len(ids))
    logger.info("Extraction done: %s new files, %s ids total", saved, len(ids))
    return len(ids)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_extraction()
