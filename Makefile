.PHONY: help install dev test test-cov lint format typecheck clean benchmark load-test docker-up docker-down

# ─── Colours ──────────────────────────────────────────────────────────────────
BLUE  := \033[34m
RESET := \033[0m

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "$(BLUE)%-18s$(RESET) %s\n", $$1, $$2}'

# ─── Setup ────────────────────────────────────────────────────────────────────
install:  ## Install runtime & dev dependencies
	pip install -r requirements.txt -r requirements-dev.txt

# ─── Development ──────────────────────────────────────────────────────────────
dev:  ## Run dev server with hot-reload
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000 --log-level debug

# ─── Testing ──────────────────────────────────────────────────────────────────
test:  ## Run test suite
	pytest tests/ -v --tb=short

test-cov:  ## Run tests with coverage report
	pytest tests/ -v --tb=short \
	  --cov=src \
	  --cov-report=term-missing \
	  --cov-report=html:htmlcov \
	  --cov-fail-under=80

test-unit:  ## Run only unit tests
	pytest tests/unit/ -v

test-integration:  ## Run only integration tests
	pytest tests/integration/ -v

# ─── Code quality ─────────────────────────────────────────────────────────────
lint:  ## Lint with ruff
	ruff check src/ tests/ scripts/

format:  ## Format with black + ruff
	black src/ tests/ scripts/
	ruff check --fix src/ tests/ scripts/

typecheck:  ## Run mypy
	mypy src/ --ignore-missing-imports

check: lint typecheck  ## Run all static checks

# ─── Benchmarking ─────────────────────────────────────────────────────────────
benchmark:  ## Run latency benchmark (100 runs)
	python scripts/benchmark_latency.py --runs 100 --output data/benchmarks/latest.json

load-test:  ## Run Locust load test against running server
	locust -f scripts/load_test.py --host http://localhost:8000 --users 20 --spawn-rate 2 --run-time 60s --headless

# ─── Docker ───────────────────────────────────────────────────────────────────
docker-up:  ## Start full stack (API + Prometheus + Grafana)
	docker compose up --build -d

docker-down:  ## Stop all containers
	docker compose down -v

docker-logs:  ## Tail API container logs
	docker compose logs -f api

# ─── Utilities ────────────────────────────────────────────────────────────────
generate-samples:  ## Generate synthetic test audio clips
	python scripts/generate_test_audio.py --output data/samples/

clean:  ## Remove caches and artefacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name ".coverage" -delete
