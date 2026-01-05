FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies required by OpenCV / Ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Safe default runtime configuration (can be overridden by Koyeb env vars)
ENV DEVICE=cpu \
    IMG_SIZE=1024 \
    MAX_IMAGE_EDGE=2048 \
    MAX_UPLOAD_MB=25 \
    MODEL_DOWNLOAD_TIMEOUT=120 \
    MODEL_PATH=models/best.pt \
    CORS_ALLOW_ORIGINS="*"

EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
