FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install deps first for layer caching. The agent's dependencies are core
# (main `dependencies`, not an extra) since P6-S5.
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-install-project --no-dev

COPY . .
RUN uv sync --no-dev

EXPOSE 8080
CMD ["uvicorn", "expenso_assistant.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
