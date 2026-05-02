# ── Build stage ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements-api.txt

# ── Runtime stage ──────────────────────────────────────────────────────────
FROM python:3.11-slim

# Non-root user for security
RUN useradd -m -u 1000 appuser
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY api/         ./api/
COPY src/         ./src/
COPY configs/     ./configs/

# Mount points for model artefacts (injected at runtime via bind-mount or volume)
RUN mkdir -p /app/models /app/data/processed

ENV MODELS_DIR=/app/models \
    PROC_DIR=/app/data/processed \
    PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "30"]
