import logging
import time
from typing import AsyncIterator, Iterator, Optional

import httpx

from .exceptions import (
    LumaAuthError,
    LumaConflictError,
    LumaError,
    LumaForbiddenError,
    LumaNotFoundError,
)

log = logging.getLogger("luma.http")

_STATUS_MAP = {
    401: LumaAuthError,
    403: LumaForbiddenError,
    404: LumaNotFoundError,
    409: LumaConflictError,
}


def _raise(resp: httpx.Response) -> None:
    if resp.status_code < 400:
        return
    try:
        msg = resp.json().get("message", resp.text)
    except Exception:
        msg = resp.text
    exc_cls = _STATUS_MAP.get(resp.status_code)
    if exc_cls:
        raise exc_cls(msg)
    raise LumaError(resp.status_code, msg)


def _decode(resp: httpx.Response):
    if not resp.content:
        return None
    ct = resp.headers.get("content-type", "")
    return resp.json() if "application/json" in ct else resp.text


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

    def get(self, path: str, params=None):
        return self._req("GET", path, params=params)

    def post(self, path: str, body=None):
        return self._req("POST", path, json=body)

    def put(self, path: str, body=None):
        return self._req("PUT", path, json=body)

    def delete(self, path: str):
        return self._req("DELETE", path)

    def stream(self, path: str, params=None):
        return self._sync.stream("GET", self._url(path), params=params)

    def _req(self, method: str, path: str, **kwargs):
        t0 = time.monotonic()
        resp = self._sync.request(method, self._url(path), **kwargs)
        ms = int((time.monotonic() - t0) * 1000)
        log.debug("%s %s -> %d (%dms)", method, self._url(path), resp.status_code, ms)
        _raise(resp)
        return _decode(resp)

    # ── async ────────────────────────────────────────────────────────────────

    async def aget(self, path: str, params=None):
        return await self._areq("GET", path, params=params)

    async def apost(self, path: str, body=None):
        return await self._areq("POST", path, json=body)

    async def aput(self, path: str, body=None):
        return await self._areq("PUT", path, json=body)

    async def adelete(self, path: str):
        return await self._areq("DELETE", path)

    def astream(self, path: str, params=None):
        return self._async.stream("GET", self._url(path), params=params)

    async def _areq(self, method: str, path: str, **kwargs):
        t0 = time.monotonic()
        resp = await self._async.request(method, self._url(path), **kwargs)
        ms = int((time.monotonic() - t0) * 1000)
        log.debug("async %s %s -> %d (%dms)", method, self._url(path), resp.status_code, ms)
        _raise(resp)
        return _decode(resp)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._sync.close()

    async def aclose(self) -> None:
        await self._async.aclose()
