app:
	uv run chainlit run src/app/main.py

app-debug:
	uv run chainlit run src/app/main.py -d

test:
	uv run pytest -v --cov=src --cov-report=html --cov-report=term

ruff:
	uv run ruff check src --fix --select I

mypy:
	uv run mypy src