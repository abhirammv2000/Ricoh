"""
src/llm_factory.py - Abstracted LLM initialisation.

Provides a single ``get_llm()`` entry-point that returns a
LangChain-compatible chat model.  The concrete provider is
selected via the ``provider`` argument or the
``DEFAULT_LLM_PROVIDER`` setting in config.py.

Supported providers
───────────────────
• ``"anthropic"`` → ChatAnthropic (requires ANTHROPIC_API_KEY)
• ``"openai"``    → placeholder for future use
• ``"google"``    → placeholder for future use
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from src.config import DEFAULT_LLM_PROVIDER


def response_text(response: Any) -> str:
    """Extract the plain text from a LangChain chat response.

    ``AIMessage.content`` is a plain ``str`` on most models, but becomes a
    **list of typed blocks** (``thinking`` / ``text`` / ``tool_use``) on any
    model that returns thinking blocks — which includes models where
    thinking is enabled by default (e.g. claude-opus-5).  Calling
    ``.strip()`` directly on that list raises ``AttributeError``, so every
    call site goes through this helper instead.

    Thinking blocks are dropped; only ``text`` content is returned.
    """
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts).strip()

    return str(content).strip()


# ── Default model names per provider ──
# NOTE: the original claude-sonnet-4-20250514 was retired on 2026-06-15
# and now 404s. claude-sonnet-4-6 is the current Sonnet (bare alias, no
# date suffix). We stay on Sonnet (not Opus) deliberately: the pipeline
# makes ~4 LLM calls per question, so Sonnet's lower cost matters, and
# temperature=0.0 is still supported on Sonnet 4.6.
_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o",
    "google": "gemini-1.5-pro",
}

# ── Models that removed the sampling parameters ────────────────────
# temperature / top_p / top_k were removed on the Opus 4.7+ and Sonnet 5
# generations: sending them returns a 400.  Sonnet 4.6 and Opus 4.6 still
# accept them.  We therefore apply `temperature` conditionally rather than
# unconditionally — passing temperature=0.0 to e.g. claude-opus-5 (which we
# use as the eval judge) would fail the request outright.
#
# Note also that temperature=0.0 never guaranteed identical outputs on any
# model; it reduces variance, it does not make sampling deterministic.
_NO_SAMPLING_PARAMS: frozenset[str] = frozenset(
    {
        "claude-opus-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-sonnet-5",
        "claude-fable-5",
        "claude-mythos-5",
    }
)

# ChatAnthropic defaults max_tokens to 1024, which is tight for a
# step-by-step procedural answer and can silently truncate mid-sentence.
# It also has to cover thinking tokens on models where thinking is on by
# default (e.g. claude-opus-5), since max_tokens caps thinking + text.
_DEFAULT_MAX_TOKENS: int = 4096


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    **kwargs,
) -> BaseChatModel:
    """Return a LangChain-compatible chat model.

    Args:
        provider:    ``"anthropic"``, ``"openai"``, or ``"google"``.
                     Falls back to ``DEFAULT_LLM_PROVIDER``.
        model:       Model identifier override.  If *None*, uses the
                     sensible default for the chosen provider.
        temperature: Sampling temperature, applied only on models that
                     still accept it (see ``_NO_SAMPLING_PARAMS``).  Low
                     temperature reduces variance; it does **not** make
                     output deterministic.
        max_tokens:  Output cap.  On models with thinking enabled by
                     default this budget covers thinking *and* text.
        **kwargs:    Forwarded to the underlying model constructor.

    Returns:
        A ``BaseChatModel`` instance ready for ``.invoke()``.

    Raises:
        ValueError:           Unknown provider string.
        NotImplementedError:  Provider not yet wired up.
    """
    provider = (provider or DEFAULT_LLM_PROVIDER).lower()
    model = model or _DEFAULT_MODELS.get(provider)

    # ── Anthropic (primary provider for this hackathon) ────────────
    if provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not found.  Add it to your .env file:\n"
                "  ANTHROPIC_API_KEY=sk-ant-..."
            )

        from langchain_anthropic import ChatAnthropic  # lazy import

        kwargs.setdefault("max_tokens", max_tokens)
        # Only send `temperature` to models that still accept it — newer
        # models reject sampling parameters with a 400.
        if model not in _NO_SAMPLING_PARAMS:
            kwargs.setdefault("temperature", temperature)

        return ChatAnthropic(model=model, **kwargs)

    # ── OpenAI (stub - activate when needed) ──────────────────────
    elif provider == "openai":
        raise NotImplementedError(
            "OpenAI provider not yet wired up. "
            "Install langchain-openai and add OPENAI_API_KEY."
        )

    # ── Google (stub - activate when needed) ──────────────────────
    elif provider == "google":
        raise NotImplementedError(
            "Google provider not yet wired up. "
            "Install langchain-google-genai and add GOOGLE_API_KEY."
        )

    else:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            "Supported: 'anthropic', 'openai', 'google'."
        )
