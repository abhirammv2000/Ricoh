# ── Citera container ──────────────────────────────────────────
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

# Bake the ONNX MiniLM embedding model into the image.
#
# ChromaDB lazily downloads it (~79MB) on the FIRST embedding call, which in a
# container means the first user request pays for a network download — and
# fails outright in an environment without egress. Fetching it at build time
# makes cold starts predictable and the container self-contained.
RUN python -c "from chromadb.utils import embedding_functions as ef; ef.ONNXMiniLM_L6_V2()(['warm up'])"

# Application code.
#
# The 223MB of source PDFs are NOT baked in (see .dockerignore), and the
# container therefore cannot build an index at boot: it would come up
# "healthy" and refuse every question, because the synthesizer correctly
# declines to answer with no evidence. So a small pre-built index ships
# instead. Rebuild it before building the image if the corpus changes:
#
#   python -m src.build_demo_index
#
COPY src/ ./src/
COPY app/ ./app/
COPY api/ ./api/
COPY eval/ ./eval/
COPY demo_index/ ./demo_index/

# Serve the baked subset, and have the UI disclose that it IS a subset —
# the published metrics were measured on the full 733-document corpus.
ENV CHROMA_DIR=/app/demo_index \
    DEMO_MODE=true

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen('http://localhost:'+os.getenv('PORT','8501')+'/_stcore/health')" || exit 1

# Respect $PORT (Render/Cloud Run set it); default to 8501 locally.
CMD ["sh", "-c", "streamlit run app/main.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
