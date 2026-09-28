up:        ; docker compose up -d --build
down:      ; docker compose down
logs:      ; docker compose logs -f
app:       ; docker compose up -d --build app mongo
pipeline:  ; docker compose run --rm app python -m src.pipeline
test:      ; pytest -q
