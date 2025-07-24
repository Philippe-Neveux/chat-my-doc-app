FROM ghcr.io/astral-sh/uv:0.7.12-debian-slim

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Default environment variable for the port
ENV PORT="8000"

ADD . /app

WORKDIR /app
RUN uv sync --locked

EXPOSE ${PORT}

CMD ["sh", "-c", "uv run chainlit run src/app/main.py --port ${PORT}"]