"""
queue.py — Durable queues (``/v1/queue``).

Delivery is **at-least-once**: :meth:`QueueClient.receive` leases messages for
a visibility window, and a message that is not acked before the window expires
becomes visible again with its ``attempts`` counter incremented. Consumers must
therefore be idempotent — the same message can legitimately arrive twice.

Typical loop::

    for msg in db.queue.receive("jobs", max=10, visibility_secs=60)["messages"]:
        handle(msg["body"])
        db.queue.ack("jobs", msg["id"])
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from ._http import Http


class QueueClient:
    """Durable, disk-backed queues."""

    def __init__(self, http: Http):
        self._http = http

    def enqueue(self, queue: str, body: Any, delay_secs: Optional[int] = None) -> dict:
        """Append a message. ``body`` is arbitrary JSON, returned verbatim."""
        return self._http.post(f"/v1/queue/{queue}", _enqueue_body(body, delay_secs))

    async def aenqueue(
        self, queue: str, body: Any, delay_secs: Optional[int] = None
    ) -> dict:
        return await self._http.apost(
            f"/v1/queue/{queue}", _enqueue_body(body, delay_secs)
        )

    def receive(
        self,
        queue: str,
        max: Optional[int] = None,
        visibility_secs: Optional[int] = None,
    ) -> dict:
        """Lease up to ``max`` messages for ``visibility_secs``.

        Returns ``{"messages": [...]}``, possibly empty. Each message carries
        ``attempts``; a value above 1 means a previous lease expired without an
        ack, which is the signal to watch for a poison message.
        """
        return self._http.post(
            f"/v1/queue/{queue}/receive", _receive_body(max, visibility_secs)
        )

    async def areceive(
        self,
        queue: str,
        max: Optional[int] = None,
        visibility_secs: Optional[int] = None,
    ) -> dict:
        return await self._http.apost(
            f"/v1/queue/{queue}/receive", _receive_body(max, visibility_secs)
        )

    def ack(self, queue: str, id: str) -> dict:
        """Delete a leased message. Without this it is redelivered."""
        return self._http.delete(f"/v1/queue/{queue}/{id}")

    async def aack(self, queue: str, id: str) -> dict:
        return await self._http.adelete(f"/v1/queue/{queue}/{id}")

    def stats(self, queue: str) -> dict:
        """Queue depth (all messages) and visible count (available to receive)."""
        return self._http.get(f"/v1/queue/{queue}")

    async def astats(self, queue: str) -> dict:
        return await self._http.aget(f"/v1/queue/{queue}")


def _enqueue_body(body: Any, delay_secs: Optional[int]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"body": body}
    if delay_secs is not None:
        payload["delay_secs"] = delay_secs
    return payload


def _receive_body(max: Optional[int], visibility_secs: Optional[int]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if max is not None:
        payload["max"] = max
    if visibility_secs is not None:
        payload["visibility_secs"] = visibility_secs
    return payload
