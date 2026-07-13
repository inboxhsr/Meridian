# ============================================================
# Meridian — Dockerfile
# Python 3.11 base with FastAPI + Streamlit
# ============================================================

FROM python:3.11-slim

WORKDIR /app

# System dependencies for whisper (ffmpeg) and pdf/image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    tesseract-ocr \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt streamlit>=1.38.0

# Application code
COPY . .

# Load env (runtime, not baked into image)
# .env must be mounted or provided at container start

EXPOSE 8000 8501

# Default: run FastAPI (Streamlit started as separate service in compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
