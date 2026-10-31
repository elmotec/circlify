.DEFAULT_GOAL := help

.PHONY: help install lint format test test-all coverage dist upgrade release clean

PYTHON_VERSIONS = $(shell python3 scripts/supported_pythons.py)

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "%-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Sync the locked development environment
	uv sync --locked

lint: ## Check lint and formatting
	uv run --locked ruff check .
	uv run --locked ruff format --check .

format: ## Apply lint fixes and formatting
	uv run --locked ruff check --fix .
	uv run --locked ruff format .

test: ## Run the unit tests
	uv run --locked python -m unittest -v

test-all: ## Test with every supported Python version
	@for version in $(PYTHON_VERSIONS); do \
		uv run --locked --python $$version python -m unittest -v || exit $$?; \
	done

coverage: ## Run tests with coverage and write coverage.xml
	uv run --locked coverage run --branch --source=circlify -m unittest
	uv run --locked coverage report -m
	uv run --locked coverage xml

dist: ## Build source and wheel distributions
	uv build

upgrade: ## Upgrade locked dependencies and pre-commit hooks
	uv lock --upgrade
	uv run --locked pre-commit autoupdate

release: ## Bump the version and update the changelog
	uv run --locked cz bump --changelog --retry ${RELEASE_OPTIONS}

clean: ## Remove generated build, test, and coverage files
	rm -rf build dist htmlcov .pytest_cache .coverage coverage.xml
