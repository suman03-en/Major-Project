FROM python:3.13-slim

# Install system dependencies (Tesseract OCR & Nepali language pack)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-nep \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy uv binary from the official Astral uv image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy requirements file and install dependencies using uv
COPY requirements.txt .
RUN uv pip install --system --no-cache-dir -r requirements.txt --index-strategy unsafe-best-match

COPY . .

COPY entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

