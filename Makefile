up:        ; docker compose up -d --build
down:      ; docker compose down
logs:      ; docker compose logs -f

# --- Run project scripts inside Docker (no local Python setup needed) ---
shell:     ; docker compose run --rm python bash
extract:   ; docker compose run --rm python python -m src.extraction.extract
inspect:   ; docker compose run --rm python python -m src.extraction.inspect_raw
clean:     ; docker compose run --rm python python -m src.cleaning.clean
mongo:     ; docker compose run --rm python python -m src.database.mongo
queries:   ; docker compose run --rm python python -m src.database.queries
pipeline:  ; docker compose run --rm python python -m src.pipeline
test:      ; docker compose run --rm python pytest -q
