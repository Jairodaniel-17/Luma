"""
stream.py — SSE streaming client for /v1/stream (formerly /v1/events).

Sync usage:
    for event in luma.stream.subscribe(since=0):
        print(event)

Async usage (idiomatic — preferred for async code):
    async for event in luma.stream.astream(since=0):
        print(event)

The legacy asubscribe() method returns an async generator for backward
compatibility, but astream() is the recommended async entrypoint.
"""
from __future__ import annotations

import json
import logging
from typing import AsyncIterator, Dict, Iterator, Optional

from ._http import Http
from .types import SseEvent

log = logging.getLogger("luma.stream")


class StreamClient:
    def __init__(self, http: Http):
        self._http = http

    # ── sync ─────────────────────────────────────────────────────────────────

    def subscribe(
        self,
        *,
        since: int = 0,
        types: Optional[str] = None,
        key_prefix: Optional[str] = None,
        collection: Optional[str] = None,
    ) -> Iterator[SseEvent]:
        """Synchronously iterate over live SSE events.

        Yields parsed dicts for each ``data:`` frame. Keepalive comments are
        silently dropped. The connection stays open until the caller breaks or
        the server closes it.
        """
        params = _stream_params(since, types, key_prefix, collection)
        with self._http.stream("/v1/stream", params=params) as resp:
            resp.raise_for_status()
            yield from _parse_sse_sync(resp)

    # ── async ────────────────────────────────────────────────────────────────

    async def astream(
        self,
        *,
        since: int = 0,
        types: Optional[str] = None,
        key_prefix: Optional[str] = None,
        collection: Optional[str] = None,
    ) -> AsyncIterator[SseEvent]:
        """Async generator that yields parsed SSE events.

        This is the idiomatic httpx async streaming pattern — it uses
        ``async with`` + ``async for`` on the response object directly so the
        connection is kept alive for the full lifetime of the generator.

        Usage::

            async for event in luma.stream.astream(since=0):
                print(event["type"], event)
        """
        params = _stream_params(since, types, key_prefix, collection)
        async with self._http.astream("/v1/stream", params=params) as resp:
            resp.raise_for_status()
            async for event in _parse_sse_async(resp):
                yield event

    async def asubscribe(
        self,
        *,
        since: int = 0,
        types: Optional[str] = None,
        key_prefix: Optional[str] = None,
        collection: Optional[str] = None,
    ) -> AsyncIterator[SseEvent]:
        """Backward-compatible alias for :meth:`astream`.

        .. deprecated::
            Use ``astream()`` instead — it is a proper ``async def`` generator
            that can be iterated with ``async for`` directly.
        """
        params = _stream_params(since, types, key_prefix, collection)
        async with self._http.astream("/v1/stream", params=params) as resp:
            resp.raise_for_status()
            async for event in _parse_sse_async(resp):
                yield event


# ── helpers ───────────────────────────────────────────────────────────────────


def _stream_params(
    since: int,
    types: Optional[str],
    key_prefix: Optional[str],
    collection: Optional[str],
) -> Dict[str, object]:
    params: Dict[str, object] = {"since": since}
    if types:
        params["types"] = types
    if key_prefix:
        params["key_prefix"] = key_prefix
    if collection:
        params["collection"] = collection
    return params


def _parse_sse_sync(resp) -> Iterator[SseEvent]:  # type: ignore[type-arg]
    """Parse ``data:`` frames from a synchronous httpx streaming response."""
    buf: list[str] = []
    for raw in resp.iter_lines():
        if not raw:
            if buf:
                payload = "\n".join(buf)
                event = _try_parse(payload)
                if event is not None:
                    yield event
                buf.clear()
            continue
        if raw.startswith("data: "):
            buf.append(raw[6:])
        elif raw.startswith(":"):
            log.debug("sse keepalive")
    # flush trailing frame (stream closed without final blank line)
    if buf:
        payload = "\n".join(buf)
        event = _try_parse(payload)
        if event is not None:
            yield event


async def _parse_sse_async(resp) -> AsyncIterator[SseEvent]:  # type: ignore[type-arg]
    """Parse ``data:`` frames from an async httpx streaming response."""
    buf: list[str] = []
    async for raw in resp.aiter_lines():
        if not raw:
            if buf:
                payload = "\n".join(buf)
                event = _try_parse(payload)
                if event is not None:
                    yield event
                buf.clear()
            continue
        if raw.startswith("data: "):
            buf.append(raw[6:])
        elif raw.startswith(":"):
            log.debug("sse keepalive")
    if buf:
        payload = "\n".join(buf)
        event = _try_parse(payload)
        if event is not None:
            yield event


def _try_parse(payload: str) -> Optional[SseEvent]:
    try:
        return json.loads(payload)  # type: ignore[return-value]
    except json.JSONDecodeError:
        log.debug("non-json sse frame: %s", payload)
        return None
