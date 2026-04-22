from typing import Any, Dict, Optional

from ._http import Http


class DiskAnnClient:
    def __init__(self, http: Http, collection: str):
        self._http = http
        self._base = f"/v1/vector/{collection}/diskann"

    def build(self, max_degree: Optional[int] = None, build_threads: Optional[int] = None,
              search_list_size: Optional[int] = None) -> dict:
        return self._http.post(f"{self._base}/build", _body(max_degree, build_threads, search_list_size))

    async def abuild(self, max_degree: Optional[int] = None, build_threads: Optional[int] = None,
                     search_list_size: Optional[int] = None) -> dict:
        return await self._http.apost(f"{self._base}/build", _body(max_degree, build_threads, search_list_size))

    def tune(self, max_degree: Optional[int] = None, build_threads: Optional[int] = None,
             search_list_size: Optional[int] = None) -> dict:
        return self._http.post(f"{self._base}/tune", _body(max_degree, build_threads, search_list_size))

    async def atune(self, max_degree: Optional[int] = None, build_threads: Optional[int] = None,
                    search_list_size: Optional[int] = None) -> dict:
        return await self._http.apost(f"{self._base}/tune", _body(max_degree, build_threads, search_list_size))

    def status(self) -> dict:
        return self._http.get(f"{self._base}/status")

    async def astatus(self) -> dict:
        return await self._http.aget(f"{self._base}/status")


def _body(max_degree: Optional[int], build_threads: Optional[int],
          search_list_size: Optional[int]) -> Optional[Dict[str, Any]]:
    body: Dict[str, Any] = {}
    if max_degree is not None:
        body["max_degree"] = max_degree
    if build_threads is not None:
        body["build_threads"] = build_threads
    if search_list_size is not None:
        body["search_list_size"] = search_list_size
    return body or None
