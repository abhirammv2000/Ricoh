# ── RicohLibrary container ──────────────────────────────────────────
# Python 3.11 to match the tested runtime. slim base keeps the image
# small; PyMuPDF / chromadb ship manylinux wheels so no build toolchain
# is required.
FROM python:3.11-slim

# Streamlit + ChromaDB friendly defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    ANONYMIZED_TELEMETRY=False \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Application code. Note: data/ and chroma_db/ are intentionally NOT
# baked into the image (see .dockerignore). Mount the PDFs at runtime:
#   docker run -p 8501:8501 -v "$PWD/data:/app/data" \
#       -e ANTHROPIC_API_KEY=sk-ant-... ricohlibrary
# The app ingests data/ and builds the index on first launch.
COPY src/ ./src/
COPY app/ ./app/
COPY eval/ ./eval/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen('http://localhost:'+os.getenv('PORT','8501')+'/_stcore/health')" || exit 1

# Respect $PORT (Render/Cloud Run set it); default to 8501 locally.
CMD ["sh", "-c", "streamlit run app/main.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
