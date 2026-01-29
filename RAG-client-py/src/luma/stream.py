from __future__ import annotations

import json
from typing import TYPE_CHECKING, Dict, Generator, Optional

if TYPE_CHECKING:
    from .client import Client, RustKissVDBError


class StreamAPI:
    def __init__(self, client: Client) -> None:
        self._client = client

    def events(
        self,
        *,
        since: Optional[int] = None,
        types: Optional[str] = None,
        key_prefix: Optional[str] = None,
    ) -> Generator[Dict[str, object], None, None]:
        params = {}
        if since is not None:
            params["since"] = int(since)
        if types:
            params["types"] = types
        if key_prefix:
            params["key_prefix"] = key_prefix

        with self._client.stream_request("GET", "/v1/stream", params=params) as resp:
            if resp.status_code >= 400:
                raise RustKissVDBError(self._client._error_message(resp))
            event: Dict[str, object] = {}
            for line in resp.iter_lines():
                if not line:
                    if event:
                        yield event
                        event = {}
                    continue
                text = line.decode("utf-8")
                if text.startswith("event:"):
                    event["event"] = text.split(":", 1)[1].strip()
                elif text.startswith("id:"):
                    event["id"] = text.split(":", 1)[1].strip()
                elif text.startswith("data:"):
                    payload = text.split(":", 1)[1].strip()
                    try:
                        event["data"] = json.loads(payload)
                    except json.JSONDecodeError:
                        event["data_raw"] = payload
            if event:
                yield event
