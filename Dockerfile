FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir .

COPY api/ api/
COPY configs/ configs/
COPY data/artifacts/ data/artifacts/
COPY data/processed/ data/processed/
COPY checkpoints/ checkpoints/

RUN useradd -m appuser
USER appuser

HEALTHCHECK \
  --interval=2m \
  --timeout=5s \
  --start-period=3m \
  CMD curl --fail http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "api.inference_api:app", "--host", "0.0.0.0", "--port", "8000"]
