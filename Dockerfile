# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Convexity — production image.
# The engine targets Python 3.9+, but the runtime image uses 3.11-slim for
# speed and up-to-date wheels. Builds a self-contained service that exposes the
# FastAPI API and serves the static dashboard at "/".
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS build

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Copy only what pip needs first, so the dependency layer caches across code edits.
COPY pyproject.toml requirements.txt README.md ./
COPY convexity ./convexity

# Build a wheelhouse into an isolated prefix we can copy into the final stage.
RUN pip install --upgrade pip wheel \
    && pip install --prefix=/install .

# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CONVEXITY_DATA_DIR=/app/data \
    CONVEXITY_LOG_LEVEL=INFO

# curl is used by the container HEALTHCHECK.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 convexity

WORKDIR /app

# Bring in the installed packages + console entrypoint from the build stage.
COPY --from=build /install /usr/local
COPY convexity ./convexity
COPY frontend ./frontend
COPY examples ./examples

RUN mkdir -p /app/data && chown -R convexity:convexity /app
USER convexity

EXPOSE 8000
VOLUME ["/app/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Serve the API + dashboard. Override CMD to run the CLI, e.g.:
#   docker run --rm convexity convexity scan --top-n 5
CMD ["uvicorn", "convexity.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
