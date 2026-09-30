"""
TMDB client
===========
A small helper class that talks to the TMDB API for us.

It takes care of:
  1. Authentication      -> the API key comes from the .env file (never written in the code!)
  2. Rate limiting       -> we wait a little between two requests
  3. HTTP errors         -> 401, 404, 429 (too many requests), 5xx (server problem)
  4. Empty responses     -> we return None instead of crashing
"""
import logging
import time

import requests

from src.config import MAX_RETRIES, REQUEST_DELAY, TMDB_API_KEY, TMDB_BASE_URL

logger = logging.getLogger(__name__)


class TMDBClient:
    def __init__(self, api_key=TMDB_API_KEY):
        # 1) Authentication ---------------------------------------------------
        if not api_key:
            raise ValueError(
                "TMDB_API_KEY is empty. Copy .env.example to .env and paste your key inside."
            )

        self.base_url = TMDB_BASE_URL.rstrip("/")
        self.headers = {"accept": "application/json"}
        self.default_params = {}

        # TMDB gives you 2 kinds of credentials:
        #  - "API Key" (short, ~32 characters)         -> sent as ?api_key=...
        #  - "API Read Access Token" (very long, JWT)  -> sent in the Authorization header
        # We detect which one you pasted by its length.
        if len(api_key) > 40:
            self.headers["Authorization"] = "Bearer " + api_key
        else:
            self.default_params["api_key"] = api_key

        # A Session re-uses the same connection -> faster for many requests
        self.session = requests.Session()

    # ------------------------------------------------------------------------
    def get(self, endpoint, params=None):
        """
        Send a GET request to `endpoint` (example: "/movie/550").
        Returns the JSON answer as a dict, or None if nothing usable came back.
        """
        url = self.base_url + endpoint

        # Merge default params (api key) with the params of this request
        all_params = dict(self.default_params)
        if params:
            all_params.update(params)

        for attempt in range(1, MAX_RETRIES + 1):
            # 2) Rate limiting: small pause before every request
            time.sleep(REQUEST_DELAY)

            # --- send the request (network problems can happen) -------------
            try:
                response = self.session.get(
                    url, headers=self.headers, params=all_params, timeout=15
                )
            except (requests.ConnectionError, requests.Timeout) as error:
                logger.warning("Network problem (%s). Attempt %s/%s", error, attempt, MAX_RETRIES)
                time.sleep(2 * attempt)  # wait longer after each failure
                continue

            code = response.status_code

            # --- 200 = everything is fine ------------------------------------
            if code == 200:
                try:
                    data = response.json()
                except ValueError:
                    logger.warning("Response is not valid JSON: %s", url)
                    return None
                # 4) Empty response ({} or []) -> treat as "nothing"
                if not data:
                    logger.warning("Empty response: %s", url)
                    return None
                return data

            # --- 3) HTTP errors ------------------------------------------------
            if code == 401:
                # Wrong key: retrying is useless, so we stop everything
                raise PermissionError("TMDB says 401: your API key/token is invalid.")

            if code == 404:
                logger.info("Not found (404): %s", url)
                return None

            if code == 429:
                # We sent too many requests. TMDB tells us how long to wait.
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

            # Any other code (400, 403...): log it and give up on this request
            logger.warning("Unexpected status %s for %s", code, url)
            return None

        logger.error("Giving up after %s attempts: %s", MAX_RETRIES, url)
        return None

    # ------------------------------------------------------------------------
    def discover_movies(self, page):
        """One page (20 movies) of the discover list. Returns a list of movies ([] if empty)."""
        data = self.get(
            "/discover/movie",
            params={
                "page": page,                       # pagination
                "language": "en-US",
                "sort_by": "popularity.desc",
                "vote_count.gte": 50,               # ignore movies nobody has rated
                "include_adult": "false",
            },
        )
        if data is None:
            return []
        return data.get("results", [])

    def movie_details(self, movie_id):
        """
        Full details of one movie.
        `append_to_response=keywords` adds the keywords in the SAME request,
        so we do 1 request per movie instead of 2.
        """
        return self.get(
            f"/movie/{movie_id}",
            params={"language": "en-US", "append_to_response": "keywords"},
        )
