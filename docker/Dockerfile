# Production Dockerfile for EdgeVLM-DPO On-Device Multimodal Service
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

EXPOSE 8002 9090

CMD ["uvicorn", "vlm_dpo.server.app:app", "--host", "0.0.0.0", "--port", "8002"]
