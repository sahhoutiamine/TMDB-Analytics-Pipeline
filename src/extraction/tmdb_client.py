"""TMDB API client: auth, retries, rate limiting, empty-response handling."""
import logging
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import TMDB_API_KEY, TMDB_BASE_URL

logger = logging.getLogger(__name__)


class TMDBClient:
    def __init__(self, api_key: str = TMDB_API_KEY, base_url: str = TMDB_BASE_URL,
                 min_interval: float = 0.05):
        if not api_key:
            raise ValueError("TMDB_API_KEY is missing. Set it in your .env file.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.min_interval = min_interval  # simple throttle between calls
        self._last_call = 0.0

        self.session = requests.Session()
        retry = Retry(total=5, backoff_factor=1,
                      status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=("GET",), respect_retry_after_header=True)
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def _get(self, endpoint: str, **params) -> dict | None:
        wait = self.min_interval - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        params["api_key"] = self.api_key
        try:
            resp = self.session.get(f"{self.base_url}{endpoint}", params=params, timeout=15)
            self._last_call = time.time()
            resp.raise_for_status()
            data = resp.json()
        except requests.HTTPError as e:
            logger.warning("HTTP error on %s: %s", endpoint, e)
            return None
        except (requests.RequestException, ValueError) as e:
            logger.error("Request failed on %s: %s", endpoint, e)
            return None
        return data or None

    def discover_page(self, page: int, **filters) -> list[dict]:
        data = self._get("/discover/movie", page=page, language="en-US",
                         sort_by="popularity.desc", **filters)
        return (data or {}).get("results", [])

    def movie_details(self, movie_id: int) -> dict | None:
        """Details + keywords in a single call."""
        return self._get(f"/movie/{movie_id}", language="en-US", append_to_response="keywords")
