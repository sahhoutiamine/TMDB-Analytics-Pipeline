import logging
import time

import requests

from src.config import MAX_RETRIES, REQUEST_DELAY, TMDB_API_KEY, TMDB_BASE_URL

logger = logging.getLogger(__name__)


class TMDBClient:
    def __init__(self, api_key=TMDB_API_KEY):
        if not api_key:
            raise ValueError(
                "TMDB_API_KEY is empty. Copy .env.example to .env and paste your key inside."
            )

        self.base_url = TMDB_BASE_URL.rstrip("/")
        self.headers = {"accept": "application/json"}
        self.default_params = {}

        if len(api_key) > 40:
            self.headers["Authorization"] = "Bearer " + api_key
        else:
            self.default_params["api_key"] = api_key

        self.session = requests.Session()

    def get(self, endpoint, params=None):
        url = self.base_url + endpoint

        all_params = dict(self.default_params)
        if params:
            all_params.update(params)

        for attempt in range(1, MAX_RETRIES + 1):
            time.sleep(REQUEST_DELAY)

            try:
                response = self.session.get(
                    url, headers=self.headers, params=all_params, timeout=15
                )
            except (requests.ConnectionError, requests.Timeout) as error:
                logger.warning("Network problem (%s). Attempt %s/%s", error, attempt, MAX_RETRIES)
                time.sleep(2 * attempt)
                continue

            code = response.status_code

            if code == 200:
                try:
                    data = response.json()
                except ValueError:
                    logger.warning("Response is not valid JSON: %s", url)
                    return None
                if not data:
                    logger.warning("Empty response: %s", url)
                    return None
                return data

            if code == 401:
                raise PermissionError("TMDB says 401: your API key/token is invalid.")

            if code == 404:
                logger.info("Not found (404): %s", url)
                return None

            if code == 429:
                try:
                    wait = int(response.headers.get("Retry-After", 5))
                except ValueError:
                    wait = 5
                logger.warning("Too many requests (429). Waiting %s seconds...", wait)
                time.sleep(wait)
                continue

            if code >= 500:
                logger.warning("Server error %s. Attempt %s/%s", code, attempt, MAX_RETRIES)
                time.sleep(2 * attempt)
                continue

            logger.warning("Unexpected status %s for %s", code, url)
            return None

        logger.error("Giving up after %s attempts: %s", MAX_RETRIES, url)
        return None

    def discover_movies(self, page):
        data = self.get(
            "/discover/movie",
            params={
                "page": page,
                "language": "en-US",
                "sort_by": "popularity.desc",
                "vote_count.gte": 50,
                "include_adult": "false",
            },
        )
        if data is None:
            return []
        return data.get("results", [])

    def movie_details(self, movie_id):
        return self.get(
            f"/movie/{movie_id}",
            params={"language": "en-US", "append_to_response": "keywords"},
        )
