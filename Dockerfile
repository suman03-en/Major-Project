# ── Base: CUDA 12.1 runtime on Ubuntu 22.04 ─────────────────────────────────
# Matches the PyTorch cu121 wheel. Slim runtime image keeps the layer small;
# the cuDNN/CUDA libs Surya needs come from torch itself at runtime.
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# System deps:
#   libgl1 / libglib2.0-0  — OpenCV headless runtime (Surya dependency)
#   libgomp1               — OpenMP for PyTorch CPU fallback
#   ca-certificates, curl  — needed by uv install
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Symlink python3.11 → python so scripts can call `python`
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Copy uv binary from the official Astral uv image (fast, reproducible installs)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first so Docker can cache this layer when code changes.
COPY requirements.txt .

# uv flags:
#   --system                   install into the system Python (no venv inside container)
#   --no-cache-dir             keep image lean
#   --index-strategy unsafe-best-match  allows mixing PyPI + PyTorch CUDA index
RUN uv pip install --system --no-cache-dir -r requirements.txt \
    --index-strategy unsafe-best-match

# ── Copy project source ───────────────────────────────────────────────────────
COPY . .
RUN uv pip install --system --no-cache-dir -e .

# ── Entrypoint ────────────────────────────────────────────────────────────────
COPY entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

# Env vars for Surya OCR: small batch sizes for 4 GB VRAM (RTX 3050)
ENV RECOGNITION_BATCH_SIZE=2
ENV DETECTOR_BATCH_SIZE=2
ENV TORCH_CUDA_ALLOC_CONF=expandable_segments:True

ENTRYPOINT ["/entrypoint.sh"]
