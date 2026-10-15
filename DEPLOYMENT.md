# Deployment Guide

Citera ships four deployment paths. All of them need an `ANTHROPIC_API_KEY`.
Two shapes:

- **Full corpus** (Docker with a mounted `data/`): the app ingests the 733
  Ricoh PDFs and builds the ChromaDB + BM25 index on first boot.
- **Baked demo subset** (Render blueprint, Cloud Run): the image ships the
  committed `demo_index/` (a small curated slice, see `src/build_demo_index.py`),
  so there is no corpus upload, no persistent disk, and nothing fetched over
  the network at boot. The UI discloses that it is a subset.

> **Why isn't the full index in the repo?** The corpus is ~223 MB of PDFs and
> the full index is large and machine-specific, so both are git-ignored. Only
> the small `demo_index/` is committed, and CI checks it has not drifted from
> the questions it is meant to answer (`python -m eval.ci_gate`).

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
2. Render -> **New -> Blueprint** -> select the repo.
3. Add `ANTHROPIC_API_KEY` as a secret env var. Optionally set `APP_PASSWORD`
   to gate the demo.

The blueprint bakes `demo_index/` into the image, so there is no disk to
provision and no PDF upload. To widen what the demo can answer, rebuild the
index locally (`python -m src.build_demo_index --extra N`) and push. Render
injects `$PORT`; the Dockerfile already binds to it.

---

## Option C: Streamlit Community Cloud (fastest demo)

1. Push to GitHub (without the PDFs, they exceed limits).
2. streamlit.io/cloud -> **New app** -> `app/main.py`.
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

The HTTP API (`Dockerfile.api`, see [docs/OPERATIONS.md](docs/OPERATIONS.md)) has been hardened. The Streamlit
page has not, and is for demos. What is done and what is still open:

- [x] API authentication: keys with labels, rotation by listing two, a rate limit per key, and failed-key
      throttling. The service will not start without keys.
- [ ] Real authentication in front of the Streamlit app (only a shared
      `APP_PASSWORD` and a global rate limit today).
- [x] Failure handling: a request deadline, a concurrency cap that refuses fast, public error messages that
      never contain provider text, and a stream that stops the model call when the client leaves.
- [x] Readiness and liveness are separate, and a bad index keeps the service out of rotation.
- [x] The index has a manifest, and a changed embedding model, partial re-ingest or tampered file is refused.
- [x] Answers from retrieval alone when the model is down, slow or the budget is spent.
- [x] A daily spend cap (per process, see the note in OPERATIONS.md).
- [x] Metrics for service health and RAG quality, alert rules checked by `promtool`, and a runbook.
- [x] Question text out of logs, and a setting for what traces keep. JSON logs with request ids.
- [x] Retrieved chunks that look like instructions to the model are withheld.
- [x] A non-root image built from its own minimal requirements, with a health check and graceful shutdown.
- [ ] Secrets via a manager (Vault / AWS Secrets Manager), not `.env`.
- [ ] Dashboards for the metrics.
- [ ] A new document or a changed one is picked up without a manual rebuild and restart.
- [x] LLM-call caching (opt-in semantic cache) + Anthropic timeout/retry handling.
- [x] Request tracing via LangSmith (wired, opt-in) plus local JSONL and a
      per-request dashboard drill-down. Sampling live traffic into the eval set
      is still open.
- [x] A CI retrieval regression + drift gate against `demo_index`. A
      CI-*built* full-corpus index still needs the source PDFs CI does not have.
- [ ] Multi-replica: the semantic cache, the rate limits, the concurrency cap and the spend count are all
      per-process. With N containers the real ceilings are N times the settings. Move them to a shared store (Redis).
- [ ] Run it under real traffic. Nothing here has served a real user, or been deployed to a cloud.
