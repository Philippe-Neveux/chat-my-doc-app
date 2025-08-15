FROM ghcr.io/astral-sh/uv:0.7.12-debian-slim

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

ADD . /app

WORKDIR /app
RUN uv sync --locked

# Cloud Run will set PORT environment variable, default to 8000
# Gradio typically uses port 7860, but we'll use 8000 for consistency
EXPOSE 8000

# Run the Gradio app
# The PORT environment variable will be automatically used by main.py
CMD ["uv", "run", "python", "src/chat_my_doc/app.py"]
