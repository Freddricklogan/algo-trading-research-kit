.PHONY: check lint typecheck test build security
check: lint typecheck test security build
lint: contrast
contrast:
	python3 src/algo_research/shell/contrast_check.py src/algo_research/shell/exec-shell.css
	uv run ruff check . && uv run ruff format --check .
typecheck:
	uv run mypy
test:
	uv run pytest --cov --cov-report=term
security:
	uv run bandit -q -r src && uv run pip-audit --skip-editable
build:
	uv run algo-research report --out dist --seed 42 --days 1500
