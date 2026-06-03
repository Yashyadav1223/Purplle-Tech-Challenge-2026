# ═══════════════════════════════════════════════════════════
# Store Intelligence System — Application Dockerfile
# ═══════════════════════════════════════════════════════════
FROM python:3.11-slim AS base

# Prevent Python from buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies required by OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies (cached layer) ───────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application source ──────────────────────────────
COPY api/ api/
COPY analytics/ analytics/
COPY config/ config/
COPY db/ db/
COPY pipeline/ pipeline/
COPY streaming/ streaming/

# ── Create non-root user ─────────────────────────────────
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
