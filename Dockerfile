# ── Base: Slim Python 3.12 (CPU-only web/RAG runtime) ─────────────────────────
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

# System deps:
#   libgl1 / libglib2.0-0  — OpenCV/imaging runtime
#   libgomp1               — OpenMP for PyTorch CPU inference
#   ca-certificates, curl  — needed by healthchecks / HTTPS
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy uv binary from the official Astral uv image (fast, reproducible installs)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# ── Install Python dependencies (CPU-only PyTorch, no CUDA/surya-ocr) ─────────
COPY requirements-dev.txt .

ENV UV_HTTP_TIMEOUT=300 \
    UV_CONCURRENT_DOWNLOADS=2

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements-dev.txt \
    --index-strategy unsafe-best-match

# ── Copy project source ───────────────────────────────────────────────────────
COPY . .
RUN uv pip install --system --no-deps -e .

# ── Entrypoint ────────────────────────────────────────────────────────────────
COPY entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
