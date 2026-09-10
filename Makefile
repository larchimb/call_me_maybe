FILES = main.py models.py parser.py

VPATH = src/

.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	uv run python src/main.py

debug:
	uv run python -m pdb src/main.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache
	rm -rf .venv

lint:
	flake8 $(FILES)
	mypy $(FILES) --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8 $(FILES)
	mypy $(FILES) --strict
