"""
blob.py — Object storage (``/v1/blob``) and on-the-fly image transforms
(``/v1/image``).

Objects are raw bytes, so these methods bypass the JSON codec: ``get`` returns
``bytes``, not a decoded body. Bucket names and every key segment are validated
server-side against ``[A-Za-z0-9._-]``; ``.``, ``..`` and path separators in a
segment are rejected, which is what makes nested keys safe.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from ._http import Http


class BlobClient:
    """Binary object storage."""

    def __init__(self, http: Http):
        self._http = http

    def put(
        self,
        bucket: str,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> dict:
        """Store an object. The write is atomic (temp file + rename)."""
        return self._http.put_bytes(f"/v1/blob/{bucket}/{key}", data, content_type)

    async def aput(
        self,
        bucket: str,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> dict:
        return await self._http.aput_bytes(
            f"/v1/blob/{bucket}/{key}", data, content_type
        )

    def get(self, bucket: str, key: str) -> bytes:
        """Fetch an object as raw bytes."""
        return self._http.get_bytes(f"/v1/blob/{bucket}/{key}")

    async def aget(self, bucket: str, key: str) -> bytes:
        return await self._http.aget_bytes(f"/v1/blob/{bucket}/{key}")

    def delete(self, bucket: str, key: str) -> dict:
        return self._http.delete(f"/v1/blob/{bucket}/{key}")

    async def adelete(self, bucket: str, key: str) -> dict:
        return await self._http.adelete(f"/v1/blob/{bucket}/{key}")

    def list(self, bucket: str) -> dict:
        """List object keys in a bucket."""
        return self._http.get(f"/v1/blob/{bucket}")

    async def alist(self, bucket: str) -> dict:
        return await self._http.aget(f"/v1/blob/{bucket}")

    # ── image transforms ─────────────────────────────────────────────────────

    def image(
        self,
        bucket: str,
        key: str,
        *,
        w: Optional[int] = None,
        h: Optional[int] = None,
        format: Optional[str] = None,
        quality: Optional[int] = None,
    ) -> bytes:
        """Resize and/or convert an object that is already in the blob store.

        Returns the transformed bytes. The stored object is never modified, so
        this is safe to call repeatedly with different parameters.

        ``format`` is ``png`` or ``jpeg``; ``quality`` applies to jpeg only.
        """
        return self._http.get_bytes(
            f"/v1/image/{bucket}/{key}" + _query(w, h, format, quality)
        )

    async def aimage(
        self,
        bucket: str,
        key: str,
        *,
        w: Optional[int] = None,
        h: Optional[int] = None,
        format: Optional[str] = None,
        quality: Optional[int] = None,
    ) -> bytes:
        return await self._http.aget_bytes(
            f"/v1/image/{bucket}/{key}" + _query(w, h, format, quality)
        )


def _query(
    w: Optional[int], h: Optional[int], format: Optional[str], quality: Optional[int]
) -> str:
    """Build the transform query string, omitting unset parameters.

    Built here rather than passed as httpx ``params`` because the byte-returning
    helpers take a path only — an omitted parameter must be absent, not empty,
    or the server would read it as a zero dimension.
    """
    parts: Dict[str, Any] = {}
    if w is not None:
        parts["w"] = w
    if h is not None:
        parts["h"] = h
    if format is not None:
        parts["format"] = format
    if quality is not None:
        parts["quality"] = quality
    if not parts:
        return ""
    return "?" + "&".join(f"{k}={v}" for k, v in parts.items())
