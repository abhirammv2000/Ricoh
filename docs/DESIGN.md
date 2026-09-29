# Citera: design notes

Observability, data handling and modeling. These used to sit in the [README](../README.md). Moved here to keep the README short. The text is unchanged apart from relative links.

## Observability

Every request is traced: a trace id, per-stage spans (retrieval and LLM), token
counts, derived cost, and **chunk attribution**, for any stored answer you can
ask which chunks produced it, at what rank and RRF score.

```bash
python -m src.trace_view              # recent traces + total recorded spend
python -m src.trace_view --last       # full span breakdown of one request
python -m src.trace_view --slowest 3  # worst latencies
python -m src.trace_view --doc aiw00a13.pdf   # every request that used a document
```

Traces are append-only JSONL in `traces/`, no service required to run this
repo. For a hosted timeline on top of that, LangSmith tracing is wired in and
opt-in: set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` and every agent run
streams to LangSmith as well, with the local JSONL still written. It stays off
by default so reproduction never depends on an external account.

An optional semantic answer cache (also off by default) short-circuits the
synthesizer for a repeat or near-duplicate question. A hit is recorded as a
zero-cost span, so the cost dashboard shows the saving rather than hiding it, and
the threshold is set from measured cosine similarity rather than guessed.

Anthropic *prompt* caching is a different lever and was assessed as not
applicable here: every static prompt prefix in this system (the synthesizer
instructions, the judge rubric, the tool-loop system prompt) is a few hundred
tokens, well under the 1024-token minimum a cache breakpoint needs, and
everything above that minimum, the retrieved evidence, is different on every
request. Manufacturing a cacheable prefix would mean prepending ~800 tokens of
few-shot examples to the synthesizer, which is a prompt change that would
invalidate every headline number in section 7 and require re-running the
benchmark, for a saving of a few hundred cached input tokens per call. Not
worth it. The pricing table still models `cache_read` and `cache_write` so the
door is open if a future prompt grows a large reusable prefix for another
reason.

The dashboard also drills into one past request: pick a trace and see its
per-stage cost and, for the retrieval span, exactly which chunks fed the answer
at what rank and RRF score. That view was CLI-only (`src/trace_view.py`); it and
the dashboard now share `perf.format_trace` so they cannot disagree.

**Answer feedback.** Each answer has a thumbs up or down, with an optional "what went wrong" note after a thumbs down. A vote is stored by trace id in `traces/feedback.jsonl` (`src/feedback.py`), so it links back to the exact chunks behind the answer. The dashboard shows the up-rate with a 95% range. `python -m eval.feedback_candidates` turns thumbs-down questions (and, if you ask, a seeded sample of thumbs-up ones) into candidate eval cases, skipping anything already in the benchmark.

That closes the first half of the loop and no more. Every candidate comes out with `needs_label: true` and empty `expected_sources`, because the label has to come from reading the source documents, not from the system's own answer. Adding a labelled candidate to the benchmark is still a manual step, on purpose. Two other limits: no real feedback has been collected yet, so there are no results to report, and on the public Cloud Run demo the file lives on the container's disk and is lost when the instance restarts. The feedback file holds users' questions and answers, so it is gitignored like the traces.

---

## 5. Data handling and preprocessing

- Dataset: official Ricoh ProcessDirector (RPD) documentation, 733 PDFs (~223 MB), stored in `data/` (gitignored due to size). Honest note: these are **individual help-topic articles**, most of which are a *single page* each, rather than a handful of 100+ page manuals. This is why nearly every citation reads "Page 1", and it means the real retrieval challenge here is **picking the right document out of 733**, not pinpointing a page within a long manual. The page-level citation machinery still works (and would matter for true multi-page manuals), but we call out the corpus shape rather than overstate it.
- **Extraction:** PyMuPDF extracts raw text page-by-page, preserving `source_document` and `page_number` metadata throughout.
- **Vision-augmented extraction ([`src/vision_ingest.py`](../src/vision_ingest.py)):** plain text extraction is blind to whatever a screenshot, diagram, or table on a page actually shows. A scan of the corpus flagged 116 pages with an embedded image, but 77% of those (146/190 images) turned out to be decorative icons under 50px, tiny bullet/toggle/note glyphs a small pilot confirmed produce useless descriptions ("a green toggle switch icon"). Filtering to images with a long edge >= 150px narrows this to the real candidate set: 31 pages across 28 documents, all screenshots, workflow diagrams, or embedded code samples. Each gets rendered and described by `claude-sonnet-5` (dialog titles, field/button labels, table values, diagram nodes and branch logic, in the order a technician reading the page would encounter them), and the description is appended to that page's text as its own labeled section before chunking, so a citation to "page N" never implies the diagram content came from the text layer. Cost: ~$1 total for all 31 pages, one-time (cached by document+page in `data/vision_cache.json`, never re-paid on a re-ingest). Verified, not assumed: a query for a value that exists only in an embedded XML sample screenshot (`aiw_OrderPropMap.pdf`) returned "information unavailable" before this change and the exact correct value, cited, after it; the curated 10-question benchmark was re-run afterward and showed no regression (`eval/metrics_postvision.json`).
- **Chunking strategy:** Sliding window of ~500 words with 50-word overlap. Word-based (not character-based) to keep semantic coherence. Overlap ensures no answer is lost at chunk boundaries.
- **Tokenisation (BM25):** Simple lowercase whitespace split - intentionally basic because error codes like `SC542` don't benefit from stemming.
- **Storage:** ChromaDB (vector index) + pickled BM25 (keyword index), both persisted to `chroma_db/` for fast restarts.
- **Limitations:** Most pages with a diagram also describe the same workflow in prose nearby, so the vision pass is a real but narrow win, not a broad retrieval lift; the corpus has no true tables (only two-column property-reference lists, which extract fine as text), so table parsing specifically was never the actual gap it looked like from the outside.

---

## 6. Modeling and AI strategy

### LLM: Claude Sonnet (Anthropic)
- **Why:** Strong instruction-following, reliable JSON output for the planner, low hallucination rate, and cheap enough to run 4-5 calls per question.
- **On temperature:** set to 0.0 to *reduce* output variance. It does **not** make generation deterministic. Temperature 0 has never guaranteed identical outputs. Measured run-to-run variation in the planner's sub-queries is the main source of end-to-end variance in this system; retrieval itself is bit-identical across runs.
- **Other providers:** `src/llm_factory.py` wires OpenAI and Gemini (the latter via its OpenAI-compatible endpoint) behind the same interface; `LLM_PROVIDER=google pip install -r requirements-providers.txt` and a `GEMINI_API_KEY` is all it takes to run the agent on Gemini. `eval/provider_bakeoff.py` compares them against a shared retrieval and one fixed judge. Judged result ([§7](EVALUATION.md#7-evaluation-and-metrics)): `gemini-3.6-flash` holds answer quality at 1/50th the cost per query, `gpt-4o-mini` drops correctness 0.21. So the model matters, not just the price. The judge is pinned to Anthropic regardless of the agent provider, which actually makes the agent/judge pairing *more* independent when the agent is not Claude.

### Prompt Engineering (4 specialised prompts)
1. **Planner prompt:** Decomposes queries into sub-queries + extracts entities. Outputs structured JSON. Includes retry-aware context injection.
2. **Verifier prompt:** Binary SUFFICIENT/INSUFFICIENT verdict. Defaults to SUFFICIENT on ambiguous output to prevent infinite loops.
3. **Synthesizer prompt:** Strict citation rules - every claim must cite `[Document Name, Page X]`. Refuses to answer when evidence is missing.
4. **Retry context:** On INSUFFICIENT verdict, the Planner receives a list of already-searched sources to broaden the next search.

### Retrieval Strategy
- **Semantic:** ChromaDB with local all-MiniLM-L6-v2 embeddings (no API key needed).
- **Keyword:** BM25Okapi over full chunk corpus.
- **Fusion:** RRF(k=60) merges rank positions, returning top-5 fused results per sub-query.

### Hallucination Control
- The Synthesizer is instructed to say *"Information unavailable in provided documents"* when evidence is insufficient - validated in our evaluation (see Section 7).
