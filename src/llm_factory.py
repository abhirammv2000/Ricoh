"""get_llm() returns a LangChain chat model for a provider.

    anthropic   ChatAnthropic, needs ANTHROPIC_API_KEY
    openai      ChatOpenAI, needs OPENAI_API_KEY
    google      Gemini through its OpenAI-compatible endpoint, needs GEMINI_API_KEY (or GOOGLE_API_KEY).
                Uses the openai client because langchain-google-genai pulls in protobuf 6, which breaks streamlit.
    self_hosted a vLLM server running the fine-tuned model, needs SELF_HOSTED_LLM_BASE_URL (vLLM ignores the key)

Anthropic is the production provider. The others are only for the provider bakeoff. The provider
comes from the argument, or DEFAULT_LLM_PROVIDER.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from src.config import DEFAULT_LLM_PROVIDER


def response_text(response: Any) -> str:
    """The plain text of a chat response.

    On models that think by default (claude-opus-5) content is a list of blocks, and .strip() on it
    raises, so everything goes through here. Thinking blocks are dropped.
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


# default model per provider. claude-sonnet-4-20250514 was retired and now 404s. Sonnet over Opus
# because of price, and Sonnet 4.6 still accepts temperature=0.
_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o-mini",
    "google": "gemini-3.6-flash",
    "self_hosted": "citera-finetuned",
}

# gemini has an openai-compatible endpoint, so one client covers both non-Anthropic providers
_GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# these models reject temperature, top_p and top_k with a 400 (Opus 4.7 and later, Sonnet 5), so
# don't send it. Temperature 0 never made output deterministic anyway.
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

# ChatAnthropic's default of 1024 can cut a step-by-step answer off. On models that think, max_tokens
# also has to cover the thinking.
_DEFAULT_MAX_TOKENS: int = 4096

# retries and timeout, so a 429, a 5xx or a dropped connection doesn't become a crash. The SDK
# does the retrying (it honours Retry-After), and each call is stateless so retrying is safe. The
# default timeout is effectively none, so a hung socket would block forever and no retry would fire.
_DEFAULT_TIMEOUT_SECONDS: float = 60.0
_DEFAULT_MAX_RETRIES: int = 3


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    **kwargs,
) -> BaseChatModel:
    """A LangChain chat model for the provider and model (both default from config).

    temperature is only sent to models that still accept it, and kwargs go straight to the
    constructor. Raises ValueError for an unknown provider.
    """
    provider = (provider or DEFAULT_LLM_PROVIDER).lower()
    model = model or _DEFAULT_MODELS.get(provider)

    if provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not found.  Add it to your .env file:\n"
                "  ANTHROPIC_API_KEY=sk-ant-..."
            )

        from langchain_anthropic import ChatAnthropic

        kwargs.setdefault("max_tokens", max_tokens)
        # setdefault so a caller or a test can still override these
        kwargs.setdefault("timeout", _DEFAULT_TIMEOUT_SECONDS)
        kwargs.setdefault("max_retries", _DEFAULT_MAX_RETRIES)
        if model not in _NO_SAMPLING_PARAMS:
            kwargs.setdefault("temperature", temperature)

        return ChatAnthropic(model=model, **kwargs)

    elif provider in ("openai", "google", "self_hosted"):
        from langchain_openai import ChatOpenAI

        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = None
            key_name = "OPENAI_API_KEY"
            required = api_key
        elif provider == "google":
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            base_url = _GEMINI_OPENAI_BASE_URL
            key_name = "GEMINI_API_KEY"
            required = api_key
        else:
            # vLLM doesn't check the key, but ChatOpenAI wants one, so any value works
            api_key = "not-needed"
            base_url = os.getenv("SELF_HOSTED_LLM_BASE_URL")
            key_name = "SELF_HOSTED_LLM_BASE_URL"
            required = base_url
        if not required:
            raise EnvironmentError(f"{key_name} not found. Add it to your .env file.")

        kwargs.setdefault("max_tokens", max_tokens)
        kwargs.setdefault("timeout", _DEFAULT_TIMEOUT_SECONDS)
        kwargs.setdefault("max_retries", _DEFAULT_MAX_RETRIES)
        kwargs.setdefault("temperature", temperature)
        return ChatOpenAI(model=model, api_key=api_key, base_url=base_url, **kwargs)

    else:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            "Supported: 'anthropic', 'openai', 'google', 'self_hosted'."
        )
