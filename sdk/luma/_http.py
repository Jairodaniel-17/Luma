"""
_http.py — Shared httpx transport layer for the Luma SDK.

Both sync (httpx.Client) and async (httpx.AsyncClient) share the same
headers and error-mapping logic. All 4xx / 5xx responses are converted
to typed LumaError subclasses before being surfaced to callers.

Status-code mapping
-------------------
401  → LumaAuthError
403  → LumaForbiddenError
404  → LumaNotFoundError
409  → LumaConflictError
4xx  → LumaError(status, message)
5xx  → LumaError(status, message)
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import httpx

from .exceptions import (
    LumaAuthError,
    LumaConflictError,
    LumaError,
    LumaForbiddenError,
    LumaNotFoundError,
)

log = logging.getLogger("luma.http")

_STATUS_MAP: Dict[int, type] = {
    401: LumaAuthError,
    403: LumaForbiddenError,
    404: LumaNotFoundError,
    409: LumaConflictError,
}


def _raise(resp: httpx.Response) -> None:
    """Raise a typed exception for any 4xx / 5xx response."""
    if resp.status_code < 400:
        return
    try:
        body = resp.json()
        msg: str = body.get("message") or body.get("error") or resp.text
    except Exception:
        msg = resp.text or f"HTTP {resp.status_code}"
    exc_cls = _STATUS_MAP.get(resp.status_code)
    if exc_cls is not None:
        raise exc_cls(msg)  # type: ignore[call-arg]
    raise LumaError(resp.status_code, msg)


def _decode(resp: httpx.Response) -> Any:
    """Decode the response body: JSON when the content-type says so, plain text otherwise."""
    if not resp.content:
        return None
    ct = resp.headers.get("content-type", "")
    if "application/json" in ct:
        return resp.json()
    return resp.text


class Http:
    """Shared httpx session — sync and async requests with timing log and error mapping."""

    def __init__(self, base_url: str, api_key: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._sync = httpx.Client(headers=headers, timeout=timeout)
        self._async = httpx.AsyncClient(headers=headers, timeout=timeout)

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    # ── sync ─────────────────────────────────────────────────────────────────

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._req("GET", path, params=params)

    def post(self, path: str, body: Optional[Any] = None) -> Any:
        return self._req("POST", path, json=body)

    def put(self, path: str, body: Optional[Any] = None) -> Any:
        return self._req("PUT", path, json=body)

    def delete(self, path: str) -> Any:
        return self._req("DELETE", path)

    def stream(self, path: str, params: Optional[Dict[str, Any]] = None):
        """Return a sync context manager for SSE / chunked streaming."""
        return self._sync.stream("GET", self._url(path), params=params)

    def put_bytes(self, path: str, data: bytes,
                  content_type: str = "application/octet-stream") -> Any:
        """PUT a raw byte payload (blob store).

        The session default Content-Type is application/json, so it is
        overridden per request here — sending bytes under a JSON content type
        would misrepresent the body to any proxy in between.
        """
        return self._req("PUT", path, content=data,
                         headers={"Content-Type": content_type})

    def get_bytes(self, path: str) -> bytes:
        """GET a response body as raw bytes, bypassing JSON/text decoding.

        Used for objects and transformed images, where decoding to str would
        corrupt the payload.
        """
        return self._raw("GET", path)

    def _raw(self, method: str, path: str, **kwargs: Any) -> bytes:
        resp = self._sync.request(method, self._url(path), **kwargs)
        _raise(resp)
        return resp.content

    def _req(self, method: str, path: str, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        resp = self._sync.request(method, self._url(path), **kwargs)
        ms = int((time.monotonic() - t0) * 1000)
        log.debug("%s %s -> %d (%dms)", method, self._url(path), resp.status_code, ms)
        _raise(resp)
        return _decode(resp)

    # ── async ────────────────────────────────────────────────────────────────

    async def aget(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return await self._areq("GET", path, params=params)

    async def apost(self, path: str, body: Optional[Any] = None) -> Any:
        return await self._areq("POST", path, json=body)

    async def aput(self, path: str, body: Optional[Any] = None) -> Any:
        return await self._areq("PUT", path, json=body)

    async def adelete(self, path: str) -> Any:
        return await self._areq("DELETE", path)

    def astream(self, path: str, params: Optional[Dict[str, Any]] = None):
        """Return an async context manager for SSE / chunked streaming."""
        return self._async.stream("GET", self._url(path), params=params)

    async def aput_bytes(self, path: str, data: bytes,
                         content_type: str = "application/octet-stream") -> Any:
        """Async counterpart of :meth:`put_bytes`."""
        return await self._areq("PUT", path, content=data,
                                headers={"Content-Type": content_type})

    async def aget_bytes(self, path: str) -> bytes:
        """Async counterpart of :meth:`get_bytes`."""
        return await self._araw("GET", path)

    async def _araw(self, method: str, path: str, **kwargs: Any) -> bytes:
        resp = await self._async.request(method, self._url(path), **kwargs)
        _raise(resp)
        return resp.content

    async def _areq(self, method: str, path: str, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        resp = await self._async.request(method, self._url(path), **kwargs)
        ms = int((time.monotonic() - t0) * 1000)
        log.debug("async %s %s -> %d (%dms)", method, self._url(path), resp.status_code, ms)
        _raise(resp)
        return _decode(resp)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying sync httpx client."""
        self._sync.close()

    async def aclose(self) -> None:
        """Close the underlying async httpx client."""
        await self._async.aclose()

    # ── context manager support ───────────────────────────────────────────────

    def __enter__(self) -> "Http":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    async def __aenter__(self) -> "Http":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()
