.PHONY: install dev api web test lint

install:
	cd backend && python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
	cd frontend && npm install

api:
	cd backend && DATA_DIR=../data DATABASE_URL=sqlite+aiosqlite:///../data/eagle_eye.db \
		.venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

web:
	cd frontend && npm run dev

test:
	cd backend && .venv/bin/pytest -q

lint:
	cd backend && .venv/bin/ruff check app tests
