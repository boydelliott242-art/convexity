# Convexity — developer task runner.
# Usage: `make help`
.DEFAULT_GOAL := help
PY ?= python3
VENV := .venv
BIN := $(VENV)/bin
PORT ?= 8000

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

$(BIN)/python: ## Create the virtualenv
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip wheel

.PHONY: install
install: $(BIN)/python ## Install the package with dev extras
	$(BIN)/pip install -e ".[dev]"

.PHONY: test
test: ## Run the full test suite
	$(BIN)/pytest -q

.PHONY: cov
cov: ## Run tests with coverage
	$(BIN)/pytest --cov=convexity --cov-report=term-missing

.PHONY: lint
lint: ## Lint with ruff
	$(BIN)/ruff check convexity tests

.PHONY: fmt
fmt: ## Auto-fix lint issues
	$(BIN)/ruff check convexity tests --fix

.PHONY: scan
scan: ## Run a quick sample scan (60-ticker universe, top 5)
	$(BIN)/convexity scan --universe-limit 60 --top-n 5

.PHONY: serve
serve: ## Serve the API + dashboard at http://localhost:$(PORT)
	$(BIN)/uvicorn convexity.api.app:app --reload --port $(PORT)

.PHONY: sample
sample: ## Regenerate examples/sample_scan.json
	$(BIN)/python scripts/generate_sample.py

.PHONY: docker-build
docker-build: ## Build the Docker image
	docker build -t convexity:latest .

.PHONY: docker-up
docker-up: ## Run API + dashboard via docker compose
	docker compose up --build

.PHONY: clean
clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache **/__pycache__ *.egg-info build dist
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
