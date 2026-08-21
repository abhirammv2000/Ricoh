# Deployment Guide

Citera ships four deployment paths. All of them need an `ANTHROPIC_API_KEY`.
The first three also need the Ricoh PDFs in `data/` (the app ingests them and
builds the ChromaDB + BM25 index on first boot). The Cloud Run path instead
serves the small baked demo subset, so it needs no corpus upload at all.

> **Why isn't the index in the image/repo?** The corpus is ~223 MB of
> PDFs and the built index is large and machine-specific, so both are
> git-ignored and excluded from the Docker build context. You supply the
> PDFs at deploy time via a mounted volume / persistent disk.

---

## Option A: Docker (recommended, reproducible)

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

## Option B: Render.com (one-click, public URL)

A [`render.yaml`](render.yaml) blueprint is included.

1. Push the repo to GitHub.
2. Render → **New → Blueprint** → select the repo.
3. Add `ANTHROPIC_API_KEY` as a secret env var.
4. Upload the PDFs to the mounted `/app/data` disk.

Render injects `$PORT`; the Dockerfile already binds to it.

---

## Option C: Streamlit Community Cloud (fastest demo)

1. Push to GitHub (without the PDFs, they exceed limits).
2. streamlit.io/cloud → **New app** → `app/main.py`.
3. Add `ANTHROPIC_API_KEY` in **Secrets**.
4. Because Streamlit Cloud has no persistent volume for 223 MB of PDFs,
   either commit a small curated subset to `data/`, or pre-build and
   commit a compressed `chroma_db/` index. Best for a scoped demo, not
   the full corpus.

---

## Option D: Google Cloud Run (public URL, serverless)

Cloud Run runs the same container serverless and scales to zero when idle. This
path serves the baked demo subset (`DEMO_MODE`, `demo_index/`), so it needs no
PDF upload and no persistent disk: the image is self-contained.

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

gcloud run deploy citera \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi --cpu 2 \
  --min-instances 0 --max-instances 3 \
  --set-env-vars "ANTHROPIC_API_KEY=sk-ant-..."
```

`gcloud` builds the Dockerfile with Cloud Build (the 223 MB corpus and the local
index are left out by `.gitignore` / `.dockerignore`), pushes to Artifact
Registry, and deploys. Cloud Run injects `$PORT`, which the Dockerfile already
binds. Scale-to-zero means you pay only per request, and a cold start is a few
seconds because the embedding model is baked into the image.

The key is passed as a runtime env var, never baked into the image. For a
longer-lived setup, keep it in Secret Manager and use `--set-secrets` instead.
To stream traces to LangSmith from the running service, add
`--set-env-vars LANGSMITH_TRACING=true,LANGSMITH_API_KEY=...` as well.

Live demo: https://citera-634289062173.us-central1.run.app

> **Heads up on cost.** The Streamlit surface is public and not rate limited
> (the token-bucket limiter guards the FastAPI in `api/`, not the UI), so an
> open URL exposes your Anthropic spend to anyone who finds it. For a durable
> public demo, put it behind auth or a rate limit, or take it down when not in
> use with `gcloud run services delete citera --region us-central1`.

---

## Production hardening checklist

The items below are deliberately **out of scope for this build**
but are the next steps for a real deployment:

- [ ] Authentication in front of the Streamlit app (it is currently open).
- [ ] Secrets via a manager (Vault / AWS Secrets Manager), not `.env`.
- [x] LLM-call caching (opt-in semantic cache) + Anthropic timeout/retry handling.
- [x] Request tracing via LangSmith (wired, opt-in). Metrics export still open.
- [ ] Pin a rebuilt index artifact in CI rather than ingesting on boot.
