# ============================================================
# FPViber - FPV Drone RAG Assistant
# ============================================================
# Multi-stage-friendly single image. Runs Flask on 0.0.0.0:5000
# so it is reachable from the host at http://localhost:5000.
#
# Persistence (mount these on `docker run`):
#   /app/data      -> your FPV knowledge base (.txt / .pdf / .docx)
#   /app/chat.db   -> SQLite conversation memory
# ============================================================

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FPVIBER_HOST=0.0.0.0 \
    FPVIBER_PORT=5000 \
    FPVIBER_DATA_FOLDER=/app/data \
    FPVIBER_DB_PATH=/app/chat.db

# System dependencies needed by faiss-cpu (libgomp) and pypdf.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so pip cache is reused on code changes.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code.
COPY app.py database.py rag_engine.py document_loader.py chunker.py ./
COPY templates ./templates
COPY static ./static

# Create mount points. `data` is bind-mounted with the knowledge base;
# `chat.db` is bind-mounted (or volume-mounted) for persistent memory.
RUN mkdir -p /app/data && touch /app/chat.db && chmod 664 /app/chat.db

# Declare volumes so persistence is explicit even without `-v` flags.
VOLUME ["/app/data"]

EXPOSE 5000

# Run Flask directly. For production-grade use, swap to gunicorn
# (already compatible with the WSGI `app` object).
CMD ["python", "app.py"]
