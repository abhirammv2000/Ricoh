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

from langchain_core.language_models.chat_models import BaseChatModel

from src.config import DEFAULT_LLM_PROVIDER


# ── Default model names per provider ──
_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
    "google": "gemini-1.5-pro",
}


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    **kwargs,
) -> BaseChatModel:
    """Return a LangChain-compatible chat model.

    Args:
        provider:    ``"anthropic"``, ``"openai"``, or ``"google"``.
                     Falls back to ``DEFAULT_LLM_PROVIDER``.
        model:       Model identifier override.  If *None*, uses the
                     sensible default for the chosen provider.
        temperature: Sampling temperature. 0.0 = deterministic
                     (ideal for factual tech-support answers).
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

        return ChatAnthropic(
            model=model,
            temperature=temperature,
            **kwargs,
        )

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
