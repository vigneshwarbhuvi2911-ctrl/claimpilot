install:
	pip install -r requirements-dev.txt

test:
	python -m pytest

eval:
	python -m eval.run

run:
	uvicorn app.main:get_app --factory --reload --port 8000

up:
	docker compose up --build

.PHONY: install test eval run up
