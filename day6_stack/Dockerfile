# syntax=docker/dockerfile:1.7
#
# Two things are being demonstrated here, and they are independent:
#
#   1. Layer ordering — dependencies are installed from the lockfile BEFORE the
#      source is copied, so editing a .py file invalidates only the last two
#      layers. This is what makes the rebuild seconds instead of minutes.
#   2. Multi-stage — `uv` itself, the C toolchain and the build backend all stay
#      in the builder stage. The runtime image gets a virtualenv and nothing else.
#
# The cache mount is the third piece and the one people leave out: even when a
# layer IS invalidated (a dependency changed), uv re-resolves against a warm
# wheel cache that survives the layer, so the penalty is a link rather than a
# download.

ARG PYTHON_VERSION=3.12

FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS builder

# copy, not hardlink: the uv cache lives on a mount, i.e. a different filesystem
# from /app, and hardlinking across the two fails with a warning per file.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Per-architecture cache id. Without it the arm64 and amd64 builds share one
# cache directory and fight over wheels that are not interchangeable.
ARG TARGETARCH

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv,id=uv-${TARGETARCH} \
    uv sync --frozen --no-install-project --no-dev

# --- everything above this line is cached across source edits ---

COPY phase0/ ./phase0/
COPY day6_stack/ ./day6_stack/
COPY day4_search/cache/ ./day4_search/cache/

RUN --mount=type=cache,target=/root/.cache/uv,id=uv-${TARGETARCH} \
    uv sync --frozen --no-dev


FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

# The venv is relocated wholesale, so the runtime path has to match the builder
# path exactly — the shebangs inside .venv/bin are absolute.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FASTEMBED_CACHE_PATH=/models/fastembed

WORKDIR /app

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /models \
    && chown app:app /models

COPY --from=builder --chown=app:app /app /app

USER app

CMD ["python", "day6_stack/healthcheck.py"]
