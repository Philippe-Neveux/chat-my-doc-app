FROM ghcr.io/astral-sh/uv:0.7.12-debian-slim

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

ADD . /app

# Set working directory
WORKDIR /app

RUN uv sync --locked

# Expose port (Cloud Run will set PORT environment variable)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["uv", "run", "chainlit", "run", "src/app/main.py", "--host", "0.0.0.0", "--port", "8000", "--headless"]