"""LLM initialisation.

get_llm() returns a LangChain chat model for the configured provider. The
provider comes from the `provider` argument, or from DEFAULT_LLM_PROVIDER in
config.py when that argument is left out.

Providers:
    anthropic   ChatAnthropic, needs ANTHROPIC_API_KEY
    openai      not wired up yet
    google      not wired up yet
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from src.config import DEFAULT_LLM_PROVIDER


def response_text(response: Any) -> str:
    """Pull the plain text out of a LangChain chat response.

    On most models `response.content` is a string. On models that return
    thinking blocks (for example claude-opus-5, where thinking is on by
    default) it comes back as a list of typed blocks instead, and calling
    `.strip()` on that list raises AttributeError. So every call site goes
    through this helper rather than touching `.content` directly.

    Thinking blocks are dropped and only the text is returned.
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


# Default model per provider.
# claude-sonnet-4-20250514 was retired on 2026-06-15 and now 404s;
# claude-sonnet-4-6 is the current Sonnet. We stay on Sonnet instead of Opus
# on purpose: the pipeline makes about four calls per question, so the lower
# price matters, and Sonnet 4.6 still accepts temperature=0.
_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o",
    "google": "gemini-1.5-pro",
}

# Models that dropped the sampling parameters (temperature, top_p, top_k).
# Opus 4.7 and later, and Sonnet 5, reject them with a 400; Sonnet 4.6 and
# Opus 4.6 still take them. So we send temperature only on models that accept
# it. Passing temperature=0 to, say, claude-opus-5 (which we use as the eval
# judge) would fail the request outright.
#
# temperature=0 was never a promise of identical output anyway. It lowers
# variance, it does not make sampling deterministic.
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

# ChatAnthropic defaults max_tokens to 1024, which is tight for a step-by-step
# answer and can cut it off mid-sentence. On models where thinking is on by
# default this budget also has to cover the thinking tokens, since max_tokens
# caps thinking plus text.
_DEFAULT_MAX_TOKENS: int = 4096

# Transport resilience. A transient failure (a 429 rate limit, a 500 or 503
# from the provider, a dropped connection) should not reach the user as a
# crash. Two settings cover the two failure modes.
#
# max_retries retries transient errors with exponential backoff. We let the
# Anthropic SDK do this rather than writing our own loop, because the SDK
# respects the server's Retry-After header. A hand-rolled retry that ignores
# Retry-After just hammers a service that already asked us to slow down and
# makes the rate limit worse. Retrying is safe here because each call is a
# stateless completion with no side effects.
#
# timeout caps a single attempt so one hung socket cannot stall the whole
# graph. The SDK default is effectively unbounded, so without this a stuck
# connection would hang forever and no retry would ever fire.
#
# 60 seconds per attempt is plenty for a long answer but still finite, and
# three retries with backoff cover almost every transient blip without making
# the user wait minutes on a provider that is genuinely down.
_DEFAULT_TIMEOUT_SECONDS: float = 60.0
_DEFAULT_MAX_RETRIES: int = 3


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    **kwargs,
) -> BaseChatModel:
    """Return a LangChain chat model.

    Args:
        provider:    "anthropic", "openai", or "google". Defaults to
                     DEFAULT_LLM_PROVIDER.
        model:       Model id override. Uses the provider default when None.
        temperature: Sampling temperature, sent only on models that still
                     accept it (see _NO_SAMPLING_PARAMS). Low temperature
                     lowers variance; it does not make output deterministic.
        max_tokens:  Output cap. On thinking-by-default models it also has to
                     cover the thinking tokens.
        **kwargs:    Passed through to the model constructor.

    Raises:
        ValueError:           unknown provider.
        NotImplementedError:  provider recognised but not wired up yet.
    """
    provider = (provider or DEFAULT_LLM_PROVIDER).lower()
    model = model or _DEFAULT_MODELS.get(provider)

    # Anthropic is the provider we actually use.
    if provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not found.  Add it to your .env file:\n"
                "  ANTHROPIC_API_KEY=sk-ant-..."
            )

        from langchain_anthropic import ChatAnthropic  # imported lazily

        kwargs.setdefault("max_tokens", max_tokens)
        # Retry transient errors with the SDK's backoff and cap each attempt.
        # setdefault so an explicit caller or a test can still override either.
        kwargs.setdefault("timeout", _DEFAULT_TIMEOUT_SECONDS)
        kwargs.setdefault("max_retries", _DEFAULT_MAX_RETRIES)
        # Send temperature only to models that still accept it; newer models
        # reject sampling parameters with a 400.
        if model not in _NO_SAMPLING_PARAMS:
            kwargs.setdefault("temperature", temperature)

        return ChatAnthropic(model=model, **kwargs)

    # openai: stub for now, wire up when we need it.
    elif provider == "openai":
        raise NotImplementedError(
            "OpenAI provider not yet wired up. "
            "Install langchain-openai and add OPENAI_API_KEY."
        )

    # google: stub for now, wire up when we need it.
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
