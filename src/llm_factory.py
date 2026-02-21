"""
src/llm_factory.py — Abstracted LLM initialisation.

Provides a single `get_llm()` entry-point that returns a
LangChain-compatible chat model.  The concrete provider
(Anthropic / Google) is selected via `provider` argument or
the DEFAULT_LLM_PROVIDER setting in config.py.

This module is a *stub* for Phase 1.  Full implementation
will be wired up in Phase 3 when we build the LangGraph agent.
"""

from __future__ import annotations

from src.config import DEFAULT_LLM_PROVIDER


def get_llm(provider: str | None = None, **kwargs):
    """Return a LangChain-compatible chat model.

    Args:
        provider: ``"anthropic"`` or ``"google"``.  Falls back to
                  ``DEFAULT_LLM_PROVIDER`` from config if *None*.
        **kwargs: Forwarded to the underlying model constructor
                  (e.g. ``temperature``, ``model_name``).

    Returns:
        A ``BaseChatModel`` instance.

    Raises:
        NotImplementedError: Until Phase 3 wires up the real models.
    """
    provider = (provider or DEFAULT_LLM_PROVIDER).lower()

    if provider == "anthropic":
        # Phase 3: from langchain_anthropic import ChatAnthropic
        raise NotImplementedError(
            "Anthropic provider will be implemented in Phase 3. "
            "Ensure ANTHROPIC_API_KEY is set in your .env file."
        )
    elif provider == "google":
        # Phase 3: from langchain_google_genai import ChatGoogleGenerativeAI
        raise NotImplementedError(
            "Google provider will be implemented in Phase 3. "
            "Ensure GOOGLE_API_KEY is set in your .env file."
        )
    else:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            "Supported: 'anthropic', 'google'."
        )
