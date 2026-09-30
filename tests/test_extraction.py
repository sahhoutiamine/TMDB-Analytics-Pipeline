import json
from unittest.mock import patch

import pytest
import requests

from src.extraction import extract
from src.extraction.tmdb_client import TMDBClient


class FakeResponse:
    def __init__(self, status=200, data=None, headers=None):
        self.status_code = status
        self._data = data
        self.headers = headers or {}

    def json(self):
        if self._data is None:
            raise ValueError("no json")
        return self._data


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def no_sleep():
    with patch("src.extraction.tmdb_client.time.sleep"):
        yield


def make_client(responses):
    client = TMDBClient(api_key="fake_key")
    client.session = FakeSession(responses)
    return client


def test_missing_key_raises():
    with pytest.raises(ValueError):
        TMDBClient(api_key="")


def test_long_token_uses_bearer_header():
    client = TMDBClient(api_key="x" * 60)
    assert "Authorization" in client.headers and "api_key" not in client.default_params


def test_success():
    client = make_client([FakeResponse(200, {"id": 1})])
    assert client.get("/movie/1") == {"id": 1}


def test_empty_response_returns_none():
    assert make_client([FakeResponse(200, {})]).get("/x") is None


def test_404_returns_none():
    assert make_client([FakeResponse(404, {})]).get("/x") is None


def test_401_raises():
    with pytest.raises(PermissionError):
        make_client([FakeResponse(401, {})]).get("/x")


def test_429_then_success():
    client = make_client([FakeResponse(429, {}, {"Retry-After": "1"}), FakeResponse(200, {"ok": 1})])
    assert client.get("/x") == {"ok": 1}


def test_500_then_success():
    client = make_client([FakeResponse(500), FakeResponse(200, {"ok": 1})])
    assert client.get("/x") == {"ok": 1}


def test_network_error_then_success():
    client = make_client([requests.ConnectionError("boom"), FakeResponse(200, {"ok": 1})])
    assert client.get("/x") == {"ok": 1}


def test_gives_up_after_max_retries():
    client = make_client([FakeResponse(500)] * 5)
    assert client.get("/x") is None


def test_pagination_stops_on_empty_page(tmp_path):
    pages = [
        FakeResponse(200, {"results": [{"id": 1}, {"id": 2}]}),
        FakeResponse(200, {"results": [{"id": 2}, {"id": 3}]}),
        FakeResponse(200, {"results": []}),
    ]
    client = make_client(pages)
    with patch.object(extract, "IDS_FILE", tmp_path / "ids.json"):
        assert extract.get_movie_ids(client, n_movies=100) == [1, 2, 3]


def test_download_saves_one_file_and_skips_existing(tmp_path):
    raw_file = tmp_path / "movies_raw.json"
    with patch.object(extract, "RAW_FILE", raw_file):
        movies = {}
        client = make_client([FakeResponse(200, {"id": 1, "title": "A"}), FakeResponse(404, {})])
        stats = extract.download_movies(client, [1, 2], movies)
        assert stats == {"downloaded": 1, "skipped": 0, "failed": 1}
        assert json.loads(raw_file.read_text())[0]["title"] == "A"

        client2 = make_client([FakeResponse(404, {})])
        stats2 = extract.download_movies(client2, [1, 2], extract.load_saved_movies())
        assert stats2["skipped"] == 1 and client2.session.calls == 1
