"""Refuse a request body that is bigger than any real question.

FastAPI reads the whole body of a request before it runs the check for an API key. With no limit, one caller with
no key at all could send hundreds of megabytes and the service would hold them in memory before saying 401. In a
test, one 300 MB request took the service from 438 MB to 1,047 MB.

So this runs first, with a limit set above the longest question that is allowed (24,013 bytes when it is 2000
emoji escaped the way most JSON clients send them). A request that says it is too big is refused without reading
the body. A request that does not say (chunked, or lying about its size) is counted as it arrives and refused the
moment it passes the limit.
"""

from __future__ import annotations

from collections.abc import Callable

from starlette.responses import JSONResponse

from api.errors import TOO_LARGE


def _declared_length(scope) -> int | None:
    for name, value in scope.get("headers", []):
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


class BodyLimitMiddleware:
    """A plain ASGI middleware, like RequestIdMiddleware, so it does not wrap streaming responses."""

    def __init__(self, app, max_bytes: int, on_refused: Callable[[str], None] | None = None) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.on_refused = on_refused

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = _declared_length(scope)
        if declared is not None and declared > self.max_bytes:
            await self._refuse(scope, receive, send)
            return

        refused = False
        seen = 0

        async def guarded_send(message) -> None:
            # once the 413 has gone out, whatever the application tries to send afterwards is dropped
            if not refused:
                await send(message)

        async def counting_receive():
            nonlocal refused, seen
            message = await receive()
            if message["type"] == "http.request" and not refused:
                seen += len(message.get("body", b""))
                if seen > self.max_bytes:
                    # FastAPI turns any error raised while it reads the body into its own 400, so the answer is
                    # sent from here, and the application is told the caller has gone, which stops it reading
                    refused = True
                    await self._refuse(scope, receive, send)
                    return {"type": "http.disconnect"}
            return message

        await self.app(scope, counting_receive, guarded_send)

    async def _refuse(self, scope, receive, send) -> None:
        if self.on_refused is not None:
            self.on_refused(TOO_LARGE.code)
        request_id = scope.get("state", {}).get("request_id", "")
        body = {"detail": TOO_LARGE.message, "code": TOO_LARGE.code, "request_id": request_id}
        # close the connection afterwards: the rest of the body is still on its way and is not worth reading. A
        # client that is still sending may see the connection reset before it reads this answer (tested: it
        # varies from run to run). Draining the upload so it always could would cost bandwidth for nothing
        await JSONResponse(body, status_code=TOO_LARGE.status, headers={"Connection": "close"})(scope, receive, send)
