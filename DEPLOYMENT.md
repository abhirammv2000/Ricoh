# 🚀 Deployment Guide

Citera ships three deployment paths. All of them need two things at
runtime: an `ANTHROPIC_API_KEY`, and the Ricoh PDFs available in `data/`
(the app ingests them and builds the ChromaDB + BM25 index on first boot).

> **Why isn't the index in the image/repo?** The corpus is ~223 MB of
> PDFs and the built index is large and machine-specific, so both are
> git-ignored and excluded from the Docker build context. You supply the
> PDFs at deploy time via a mounted volume / persistent disk.

---

## Option A — Docker (recommended, reproducible)

```bash
# Build
docker build -t citera .

# Run (mount the PDFs, pass the API key)
docker run --rm -p 8501:8501 \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -v "$PWD/data:/app/data" \
  -v "$PWD/chroma_db:/app/chroma_db" \
  citera
```

Open http://localhost:8501. Mounting `chroma_db` persists the index so
later restarts skip re-ingestion. A `/_stcore/health` healthcheck is
built in.

---

## Option B — Render.com (one-click, public URL)

A [`render.yaml`](render.yaml) blueprint is included.

1. Push the repo to GitHub.
2. Render → **New → Blueprint** → select the repo.
3. Add `ANTHROPIC_API_KEY` as a secret env var.
4. Upload the PDFs to the mounted `/app/data` disk.

Render injects `$PORT`; the Dockerfile already binds to it.

---

## Option C — Streamlit Community Cloud (fastest demo)

1. Push to GitHub (without the PDFs — they exceed limits).
2. streamlit.io/cloud → **New app** → `app/main.py`.
3. Add `ANTHROPIC_API_KEY` in **Secrets**.
4. Because Streamlit Cloud has no persistent volume for 223 MB of PDFs,
   either commit a small curated subset to `data/`, or pre-build and
   commit a compressed `chroma_db/` index. Best for a scoped demo, not
   the full corpus.

---

## Production hardening checklist

The items below are deliberately **out of scope for the hackathon build**
but are the next steps for a real deployment:

- [ ] Authentication in front of the Streamlit app (it is currently open).
- [ ] Secrets via a manager (Vault / AWS Secrets Manager), not `.env`.
- [ ] LLM-call caching + Anthropic rate-limit/retry handling.
- [ ] Request tracing (LangSmith) and metrics export.
- [ ] Pin a rebuilt index artifact in CI rather than ingesting on boot.
