.PHONY: env format lint test clean

ENV_NAME ?= kindle_ocr

env:
	./scripts/bootstrap_conda.sh $(ENV_NAME)

format:
	ruff check src tests --select I --fix

lint:
	ruff check src tests

mypy:
	mypy src

test:
	pytest

clean:
	rm -rf .ruff_cache .mypy_cache .pytest_cache
