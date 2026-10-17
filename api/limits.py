"""A cap on how many questions run at once.

Each question uses a worker thread for retrieval and holds a connection to the model provider. With no
cap, a burst makes every request slower until all of them time out. With a cap, the requests that fit stay
quick, and the rest wait briefly for a slot and are then refused with a 503 the caller can retry.
"""

from __future__ import annotations

import asyncio


class Lease:
    """One slot. Release it once, or as often as you like: a second release does nothing."""

    def __init__(self, gate: ConcurrencyGate) -> None:
        self._gate = gate
        self._released = False

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._gate._release()


class ConcurrencyGate:
    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        self.limit = limit
        self._slots = asyncio.Semaphore(limit)
        self._in_flight = 0
        self.rejected = 0

    @property
    def in_flight(self) -> int:
        return self._in_flight

    async def acquire(self, wait_seconds: float) -> Lease | None:
        """A slot, or None if none came free within wait_seconds."""
        if wait_seconds <= 0:
            # wait_for with no time can refuse a free slot, since the acquire has not run yet. With
            # nothing awaited between the check and the acquire, no one else can take the slot in between
            if self._slots.locked():
                self.rejected += 1
                return None
            await self._slots.acquire()
        else:
            try:
                await asyncio.wait_for(self._slots.acquire(), timeout=wait_seconds)
            except TimeoutError:
                self.rejected += 1
                return None
        self._in_flight += 1
        return Lease(self)

    def _release(self) -> None:
        self._in_flight -= 1
        self._slots.release()
