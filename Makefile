.PHONY: install dev test lint format clean run

install:
	uv pip install -e .

dev:
	uv pip install -e ".[dev]"

test:
	uv run python -m pytest tests/ -v --tb=short

test-cov:
	uv run python -m pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache .coverage htmlcov dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

run:
	uv run python -m src.app

check: lint test
