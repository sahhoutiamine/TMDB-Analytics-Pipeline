# Movie ML Platform (TMDB)

Extraction -> Cleaning -> MongoDB -> EDA -> Features -> TF-IDF -> Classification / Clustering / Recommendation -> Streamlit + Airflow + Docker.

## Quick start
```bash
cp .env.example .env        # add TMDB_API_KEY and an Airflow Fernet key
# Fernet key: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
docker compose up -d --build
```
| Service | URL |
|---|---|
| Streamlit | http://localhost:8501 |
| Airflow | http://localhost:8080 (admin / admin) |
| Mongo Express | http://localhost:8081 |

Run without Airflow: `docker compose run --rm app python -m src.pipeline`

Local dev: `python -m venv .venv && pip install -r requirements.txt` (set `MONGO_URI=mongodb://localhost:27017`).
