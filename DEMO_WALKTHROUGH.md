# RicohLibrary — Demo Walkthrough (5-7 min)

## 🎙️ Opening (30 seconds)

> "We built **RicohLibrary** — an agentic AI system that doesn't just search Ricoh manuals, it **reasons** through them. It plans, retrieves in two passes, verifies evidence, and only answers when it's confident — with full citations."

## 📋 Slide 1: Architecture (1 min)

**Talk through the 9-step pipeline:**

1. User asks a question
2. **Planner** breaks it into sub-queries + extracts entities (error codes, model numbers)
3. **Retriever Pass 1** — searches with sub-queries (semantic + keyword hybrid)
4. **Retriever Pass 2** — searches with entity-boosted refined queries
5. Evidence combined & de-duplicated
6. **Verifier** checks if evidence is sufficient
7. If insufficient → loops back to Planner with broader queries (max 2 iterations)
8. **Synthesizer** generates grounded answer with `[Document Name, Page X]` citations
9. If no evidence → refuses to answer (no hallucination)

**Key differentiator:** "This is NOT a simple RAG demo. It's a structured reasoning loop with verification."

---

## 💻 Live Demo (4-5 min)

### Start Streamlit
```powershell
streamlit run app/main.py
```

### Demo Question 1 — Simple (show speed + citations)
> **Type:** `What is the command to shut down RPD?`

**Point out:**
- ⏱️ Answer in ~10-14s
- 🕵️ Open the **Agent Thoughts** expander
- Show: sub-queries, entities (`RPD`), evidence cards with **document + page number**
- Highlight the citation format in the answer: `[aiw00a13.pdf, Page X]`

### Demo Question 2 — Entity-heavy (show two-pass retrieval)
> **Type:** `Does RPD work with FusionPro?`

**Point out:**
- 🏷️ Entities extracted: `RPD`, `FusionPro`
- 📚 Pass 1 retrieves sub-query results, Pass 2 adds entity-boosted results
- Terminal shows both passes with chunk counts

### Demo Question 3 — Hallucination control (crucial for rubric)
> **Type:** `How do I make a car?`

**Point out:**
- Agent correctly says **"Information unavailable"**
- Does NOT hallucinate or make up an answer
- "This satisfies the uncertainty handling requirement"

### Demo Question 4 — Multi-step (show retry loop)
> **Type:** `How do I create a workflow?`

**Point out:**
- Planner decomposes into focused sub-queries
- Evidence from multiple documents combined
- Step-by-step instructions in the answer

---

## 📊 Slide 2: Evaluation Results (30 seconds)

> "We ran all 10 official test questions. Average latency 13.9 seconds. Every answer has traceable citations. The agent correctly refused to answer questions outside the documentation."

- Show `evaluation_report.md` briefly

---

## 🔑 Closing (30 seconds)

**Three key takeaways:**
1. **Two-pass retrieval** — hybrid (semantic + BM25) with entity boosting
2. **Verify-and-retry loop** — agent checks its own work before answering
3. **Glass Box transparency** — judges can see every step of the reasoning

> "RicohLibrary doesn't just find answers — it **reasons** to them, **verifies** them, and **cites** them."
