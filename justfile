default:
    @just --list

# Install dependencies and pre-commit hooks
setup:
    uv sync
    uv run pre-commit install

# Run tests
test:
    uv run pytest

# Lint and format check
lint:
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy .

# Auto-fix lint and formatting issues
fmt:
    uv run ruff check --fix .
    uv run ruff format .

# Run all pre-commit hooks against all files
check:
    uv run pre-commit run --all-files
