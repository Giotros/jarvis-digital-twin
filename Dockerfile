# Jarvis George — FastAPI Inference Server
# Lightweight image for the orchestration API (no GPU deps here;
# model inference goes through Ollama via HTTP).

FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Application code
COPY src/ src/
COPY config/ config/

# Make the package importable
ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "jarvis.inference.api:app", "--host", "0.0.0.0", "--port", "8000"]
