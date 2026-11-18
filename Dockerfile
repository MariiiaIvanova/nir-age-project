FROM python:3.10-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/nir
COPY requirements.txt .
RUN python -m pip install -r requirements.txt && python -m pip check && python -c "import torch, torchaudio, whisper"
COPY app/ ./
