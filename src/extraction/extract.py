"""
Extraction script  (STEP 1)
===========================
Run it with:   python -m src.extraction.extract

What it does:
  A) Collect movie IDs page by page (pagination)
  B) Download the details of each movie
  C) Save EVERYTHING in one file: data/raw/movies_raw.json  (a list of movies)
  D) If a movie is already in the file, skip it -> we never call the API twice for the same movie
"""
import json
import logging

from src.config import DATA_RAW, N_MOVIES, RAW_MOVIES_FILE
from src.extraction.tmdb_client import TMDBClient

logger = logging.getLogger(__name__)

MAX_PAGES = 500                          # TMDB does not give more than 500 pages
IDS_FILE = DATA_RAW / "movie_ids.json"   # cache of the list of IDs (saves ~100 API calls)
RAW_FILE = RAW_MOVIES_FILE


# --------------------------------------------------------------------------
# Reading / writing the single JSON file
# --------------------------------------------------------------------------
def load_saved_movies():
    """Read movies_raw.json and return a dict {movie_id: movie_data}."""
    if not RAW_FILE.exists():
        return {}
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        movie_list = json.load(f)
    return {movie["id"]: movie for movie in movie_list}


def save_movies(movies):
    """
    Write all movies to movies_raw.json.
    We write to a temporary file first, then rename it: if the program crashes
    during the write, the old file is not corrupted.
    """
    tmp_file = RAW_FILE.with_suffix(".tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(list(movies.values()), f, ensure_ascii=False)
    tmp_file.replace(RAW_FILE)


# --------------------------------------------------------------------------
# A) Movie IDs
# --------------------------------------------------------------------------
def get_movie_ids(client, n_movies, force=False):
    """Return a list of `n_movies` movie IDs (uses a local cache if possible)."""
    if IDS_FILE.exists() and not force:
        saved_ids = json.loads(IDS_FILE.read_text())
        if len(saved_ids) >= n_movies:
            logger.info("IDs loaded from cache (%s ids)", len(saved_ids))
            return saved_ids[:n_movies]

    ids = []
    page = 1
    while len(ids) < n_movies and page <= MAX_PAGES:
        movies = client.discover_movies(page)

        if not movies:  # empty page -> no more results, stop
            logger.info("Page %s is empty: stopping pagination", page)
            break

        for movie in movies:
            if movie["id"] not in ids:  # avoid duplicates
                ids.append(movie["id"])

        logger.info("Page %s done -> %s ids collected", page, len(ids))
        page += 1

    ids = ids[:n_movies]
    IDS_FILE.write_text(json.dumps(ids))
    return ids


# --------------------------------------------------------------------------
# B) Movie details
# --------------------------------------------------------------------------
def download_movies(client, ids, movies, force=False):
    """
    Download the details of each movie and add them to the `movies` dict.
    The file is saved every 100 downloads, so a crash never loses much work.
    """
    downloaded = 0
    skipped = 0
    failed = 0

    for position, movie_id in enumerate(ids, start=1):
        # Already in the file -> skip
        if movie_id in movies and not force:
            skipped += 1
            continue

        details = client.movie_details(movie_id)
        if details is None:  # 404, empty answer, repeated errors...
            failed += 1
            continue

        movies[movie_id] = details
        downloaded += 1

        if downloaded % 100 == 0:
            save_movies(movies)
            logger.info("Progress: %s / %s movies (saved)", position, len(ids))

    save_movies(movies)  # final save
    return {"downloaded": downloaded, "skipped": skipped, "failed": failed}


def run_extraction(n_movies=N_MOVIES, force=False):
    """Full extraction. Returns a small dict with statistics."""
    movies = {} if force else load_saved_movies()

    client = TMDBClient()
    ids = get_movie_ids(client, n_movies, force)
    stats = download_movies(client, ids, movies, force)
    stats["total_ids"] = len(ids)
    stats["movies_in_file"] = len(movies)
    logger.info("Extraction finished: %s", stats)
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_extraction()
