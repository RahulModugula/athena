.PHONY: up down demo test lint typecheck clean

# Start the full stack (postgres, redis, backend, streamlit)
up:
	docker compose up --build

down:
	docker compose down

# Seed demo documents and open the UI
demo: up
	@echo "Waiting for backend to be ready..."
	@until curl -sf http://localhost:8000/api/health > /dev/null; do sleep 2; done
	python -m scripts.seed_demo --host http://localhost:8000
	@echo "Open http://localhost:8501 to start querying"

# Run tests
test:
	cd backend && python -m pytest tests/ -v --cov=app --cov-report=term-missing

# Lint
lint:
	cd backend && python -m ruff check .

# Type checking
typecheck:
	cd backend && python -m mypy app/

# Full CI check
ci: lint typecheck test

clean:
	docker compose down -v
	find . -type d -name __pycache__ | xargs rm -rf
	find . -name "*.pyc" -delete
