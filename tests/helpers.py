"""Reusable test doubles."""

from __future__ import annotations

from types import SimpleNamespace


class FakeLLM:
    """Minimal stand-in for a LangChain chat model.

    ``invoke(prompt)`` returns an object with a ``.content`` attribute,
    matching the interface the agent nodes rely on. The canned response
    is fixed at construction time so tests are fully deterministic.
    """

    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list[str] = []

    def invoke(self, prompt: str):  # noqa: D401 - mimics LangChain
        self.calls.append(prompt)
        return SimpleNamespace(content=self._response)
