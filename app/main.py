"""
app/main.py - Citera Streamlit Glass Box Dashboard.

This is Phase 5 of the hackathon project.  It provides a
professional chat-style interface that shows not just the
agent's answer, but its full reasoning process - the "Glass Box".

Features:
─────────
• Chat-style UI with user/assistant message bubbles
• Glass Box expander showing: sub-queries, evidence sources,
  verification status, iteration count
• Sidebar with system controls, model info, and reset button
• Clean, dark-themed styling
"""

# ── Telemetry + env MUST run before ANY other imports ──
import os
import sys

os.environ["ANONYMIZED_TELEMETRY"] = "False"

# ── Ensure project root is on sys.path so `src` package is importable ──
# Streamlit runs this file directly, so the project root isn't
# automatically on the path.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import logging
import time

import streamlit as st

# ── Ensure config.py logging setup runs ──
from src.config import (  # noqa: F401
    CHROMA_DIR,
    DEFAULT_LLM_PROVIDER,
    DEMO_MODE,
    USE_PLANNER,
)

# Suppress noisy loggers in Streamlit context
for _quiet in ("src.ingest", "src.retriever", "chromadb", "httpx", "httpcore"):
    logging.getLogger(_quiet).setLevel(logging.WARNING)

from src.agent import stream_agent, StreamResult
from src.ingest import ingest_all
from src.instrumentation import record_run
from src.retriever import HybridRetriever
from src.llm_factory import _DEFAULT_MODELS

logger = logging.getLogger(__name__)


# ====================================================================
# 1. PAGE CONFIGURATION
# ====================================================================

st.set_page_config(
    page_title="Citera",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ====================================================================
# 2. CUSTOM CSS - Premium dark Glass Box styling
# ====================================================================

st.markdown(
    """
    <style>
    /* ── Global font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Main header ── */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .main-header h1 {
        color: #e94560;
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .main-header p {
        color: #a8b2c1;
        margin: 0.3rem 0 0 0;
        font-size: 0.9rem;
    }

    /* ── Glass Box expander styling ── */
    .glass-box {
        background: rgba(22, 33, 62, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(233, 69, 96, 0.2);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
    }
    .glass-box h4 {
        color: #e94560;
        margin: 0 0 0.5rem 0;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .glass-box ul {
        margin: 0;
        padding-left: 1.2rem;
    }
    .glass-box li {
        color: #c8d0dc;
        font-size: 0.85rem;
        margin-bottom: 0.3rem;
    }

    /* ── Evidence cards ── */
    .evidence-card {
        background: rgba(15, 52, 96, 0.4);
        border-left: 3px solid #e94560;
        padding: 0.6rem 0.8rem;
        margin: 0.4rem 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.82rem;
        color: #c8d0dc;
    }
    .evidence-card strong {
        color: #e94560;
    }

    /* ── Stat pills in sidebar ── */
    .stat-pill {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(233, 69, 96, 0.3);
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin: 0.3rem 0;
        text-align: center;
    }
    .stat-pill .label {
        color: #8892a4;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stat-pill .value {
        color: #e94560;
        font-size: 1.1rem;
        font-weight: 600;
    }

    /* ── Chat message fine-tuning ── */
    .stChatMessage {
        border-radius: 10px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ====================================================================
# 3. SESSION STATE INITIALISATION
# ====================================================================

def init_session() -> None:
    """Bootstrap session state on first load."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "retriever_ready" not in st.session_state:
        st.session_state.retriever_ready = False
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0

init_session()


# ====================================================================
# 4. INDEX BOOTSTRAP (runs once)
# ====================================================================

@st.cache_resource(show_spinner="📚 Loading retrieval index…")
def get_retriever() -> HybridRetriever:
    """Load or build the hybrid retriever (cached across reruns)."""
    retriever = HybridRetriever()

    if retriever.index_size == 0 or not retriever.bm25_ready:
        chunks = ingest_all()
        if chunks:
            retriever.build_index(chunks)

    return retriever


def ensure_index() -> None:
    """Make sure the index is loaded before accepting queries."""
    if not st.session_state.retriever_ready:
        retriever = get_retriever()
        st.session_state.retriever_ready = True
        st.session_state.index_size = retriever.index_size


# ====================================================================
# 6. GLASS BOX RENDERER
# ====================================================================

def render_glass_box(state: dict, latency: float) -> None:
    """Render the Glass Box panel showing agent internals."""

    with st.expander("🕵️ Agent Thoughts & Evidence", expanded=False):

        # -- Metrics row --
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("⏱️ Latency", f"{latency:.1f}s")
        with col2:
            st.metric("🔄 Retrieval passes", state.get("iterations", "?"))
        with col3:
            # An empty verification status means the verifier stage is off,
            # not that verification failed. Say which, rather than render a
            # blank metric the viewer has to guess about.
            status = state.get("verification_status") or "off"
            st.metric("✅ Verification", status)

        # Live latency, cost, and token usage for this request, from the trace.
        trace = state.get("trace")
        if trace:
            ttft = trace.get("ttft_seconds")
            ttft_str = f"⚡ {ttft:.1f}s to first token  |  " if ttft else ""
            st.caption(
                f"{ttft_str}"
                f"💰 ${trace['cost_usd']:.4f}  |  "
                f"{trace['llm_calls']} LLM calls  |  "
                f"{trace['input_tokens']:,} in / {trace['output_tokens']:,} out tokens"
            )

        st.divider()

        # -- Plan: only meaningful when the planner stage is enabled --
        if USE_PLANNER:
            st.markdown("#### 🧠 Plan")
            st.json(
                {
                    "sub_queries": state.get("sub_queries", []),
                    "entities": state.get("entities", []),
                },
                expanded=True,
            )
        else:
            st.markdown("#### 🧠 Plan")
            st.caption(
                "Planner and verifier stages are **disabled by default**. A "
                "progressive-removal ablation found the single-retrieval path "
                "matched or beat the full agentic pipeline on every quality "
                "metric at 3.1× lower cost and 1.6× lower latency — so the "
                "query is sent to retrieval unchanged. Set `USE_PLANNER=true` "
                "/ `USE_VERIFIER=true` to re-enable. See §7 of the README."
            )

        st.divider()

        # -- Evidence: retrieved chunks (JSON formatted) --
        evidence = state.get("retrieved_evidence", [])
        st.markdown(f"#### 📚 Evidence - {len(evidence)} chunks retrieved")

        if evidence:
            evidence_display = []
            for i, e in enumerate(evidence, 1):
                evidence_display.append({
                    "index": i,
                    "source": e.get("source_document", "unknown"),
                    "page": e.get("page_number", "?"),
                    "snippet": e.get("text", "")[:200].replace("\n", " "),
                })
            st.json(evidence_display, expanded=False)
        else:
            st.caption("No evidence retrieved.")


# ====================================================================
# 7. SIDEBAR
# ====================================================================

with st.sidebar:
    st.markdown("## ⚙️ System Controls")
    st.divider()

    # ── Model info ──
    provider = DEFAULT_LLM_PROVIDER
    model_name = _DEFAULT_MODELS.get(provider, "unknown")

    st.markdown(
        f'<div class="stat-pill">'
        f'<div class="label">LLM Provider</div>'
        f'<div class="value">{provider.title()}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="stat-pill">'
        f'<div class="label">Model</div>'
        f'<div class="value">{model_name}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Index status ──
    ensure_index()
    idx_size = st.session_state.get("index_size", 0)
    st.markdown(
        f'<div class="stat-pill">'
        f'<div class="label">Index Size</div>'
        f'<div class="value">{idx_size:,} chunks</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="stat-pill">'
        f'<div class="label">Queries This Session</div>'
        f'<div class="value">{st.session_state.total_queries}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Reset button ──
    if st.button("🗑️ Reset Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.rerun()

    st.divider()
    st.caption("Citera — agentic RAG over RICOH ProcessDirector docs")
    st.caption("by Abhiram")


# ====================================================================
# 8. MAIN CHAT INTERFACE
# ====================================================================

# ── Header ──
st.markdown(
    '<div class="main-header">'
    "<h1>📚 Citera</h1>"
    "<p>Grounded technical support over RICOH ProcessDirector documentation</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ── Demo-subset disclosure ──
# The hosted demo answers from a small subset while the published metrics were
# measured on the full 733-document corpus. Letting a visitor assume the two
# are the same would overstate what they are looking at.
if DEMO_MODE:
    _n = "a reduced subset"
    try:
        import json as _json

        _manifest = CHROMA_DIR / "manifest.json"
        if _manifest.exists():
            _m = _json.loads(_manifest.read_text(encoding="utf-8"))
            _n = f"{_m['document_count']} of 733 documents"
    except Exception:  # pragma: no cover - display-only
        pass
    st.info(
        f"**Demo deployment.** This instance searches **{_n}** so the image "
        "stays small and limits redistribution of RICOH's documentation. "
        "The published evaluation metrics were measured on the **full "
        "733-document corpus** — this demo shows the system working, it does "
        "not reproduce the benchmark. Questions outside the subset will be "
        "correctly refused.",
        icon="ℹ️",
    )

# ── Render chat history ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # If this was an assistant message with agent state, show Glass Box
        if msg["role"] == "assistant" and "agent_state" in msg:
            render_glass_box(msg["agent_state"], msg.get("latency", 0))

# -- Chat input --
if user_input := st.chat_input("Ask a Ricoh technical support question..."):

    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream the answer token by token, so output appears as soon as the
    # synthesizer starts writing rather than after the whole response is ready.
    with st.chat_message("assistant"):
        t0 = time.perf_counter()
        result = StreamResult()
        try:
            with record_run(query=user_input) as rec:
                answer = st.write_stream(stream_agent(user_input, result))
            state = dict(result.final_state or {})
            state["trace"] = {
                "cost_usd": rec.total_cost_usd,
                "llm_calls": rec.llm_calls,
                "input_tokens": rec.total_input_tokens,
                "output_tokens": rec.total_output_tokens,
                "ttft_seconds": result.ttft_seconds,
            }
        except Exception as e:
            logger.error("Agent error: %s", e)
            answer = "Sorry, something went wrong while answering. Please try again."
            st.markdown(answer)
            state = {}
        latency = time.perf_counter() - t0

        if state:
            render_glass_box(state, latency)

    # Save to history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "agent_state": state,
            "latency": latency,
        }
    )
    st.session_state.total_queries += 1
