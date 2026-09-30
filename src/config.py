"""Central configuration (paths, env variables, constants)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATA_RAW = PROJECT_ROOT / "data" / "raw"
RAW_MOVIES_FILE = DATA_RAW / "movies_raw.json"  # ALL raw movies in ONE file
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
TMDB_BASE_URL = os.getenv("TMDB_BASE_URL", "https://api.themoviedb.org/3")
N_MOVIES = int(os.getenv("N_MOVIES", 2000))

# Politeness settings for the API
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", 0.1))  # seconds to wait between 2 requests
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))           # how many times we retry a failed request

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "movies_db")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "movies")

RANDOM_STATE = 42

for _p in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, FIGURES_DIR):
    _p.mkdir(parents=True, exist_ok=True)
