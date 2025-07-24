FROM ghcr.io/astral-sh/uv:0.7.12-debian-slim

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

ADD . /app

WORKDIR /app
RUN uv sync --locked

# Cloud Run will set PORT environment variable, default to 8080
EXPOSE 8080

CMD ["sh", "-c", "uv run chainlit run src/app/main.py --host 0.0.0.0 --port ${PORT:-8080} --headless"]