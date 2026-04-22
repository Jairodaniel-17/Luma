import json
import logging
from typing import AsyncIterator, Dict, Iterator, Optional

from ._http import Http

log = logging.getLogger("luma.stream")


class StreamClient:
    def __init__(self, http: Http):
        self._http = http

    def subscribe(self, *, since: int = 0, types: Optional[str] = None,
                  key_prefix: Optional[str] = None,
                  collection: Optional[str] = None) -> Iterator[dict]:
        params = _stream_params(since, types, key_prefix, collection)
        with self._http.stream("/v1/stream", params=params) as resp:
            resp.raise_for_status()
            yield from _sse_sync(resp)

    def asubscribe(self, *, since: int = 0, types: Optional[str] = None,
                   key_prefix: Optional[str] = None,
                   collection: Optional[str] = None) -> AsyncIterator[dict]:
        params = _stream_params(since, types, key_prefix, collection)
        return _sse_async(self._http, params)


def _stream_params(since: int, types: Optional[str],
                   key_prefix: Optional[str], collection: Optional[str]) -> Dict:
    params: Dict = {"since": since}
    if types:
        params["types"] = types
    if key_prefix:
        params["key_prefix"] = key_prefix
    if collection:
        params["collection"] = collection
    return params


def _sse_sync(resp) -> Iterator[dict]:
    buf: list[str] = []
    for raw in resp.iter_lines():
        if not raw:
            if buf:
                payload = "\n".join(buf)
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    log.debug("non-json sse: %s", payload)
                buf.clear()
            continue
        if raw.startswith("data: "):
            buf.append(raw[6:])
        elif raw.startswith(":"):
            log.debug("sse keepalive")


async def _sse_async(http: Http, params: Dict) -> AsyncIterator[dict]:
    async with http.astream("/v1/stream", params=params) as resp:
        resp.raise_for_status()
        buf: list[str] = []
        async for raw in resp.aiter_lines():
            if not raw:
                if buf:
                    payload = "\n".join(buf)
                    try:
                        yield json.loads(payload)
                    except json.JSONDecodeError:
                        log.debug("non-json sse: %s", payload)
                    buf.clear()
                continue
            if raw.startswith("data: "):
                buf.append(raw[6:])
            elif raw.startswith(":"):
                log.debug("sse keepalive")
