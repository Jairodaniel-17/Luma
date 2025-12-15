from __future__ import annotations

import httpx
from httpx import MockTransport, Request, Response

import unittest

from rustkissvdb import Client, RustKissVDBError


def _mock_transport() -> MockTransport:
    def handler(request: Request) -> Response:
        if request.method == "GET" and request.url.path == "/v1/vector":
            return Response(
                200,
                json={"collections": [{"collection": "docs", "dim": 3, "metric": "cosine"}]},
            )
        if request.method == "GET" and request.url.path == "/v1/vector/docs":
            return Response(
                200,
                json={"collection": "docs", "dim": 3, "metric": "cosine", "count": 0},
            )
        if request.method == "GET" and request.url.path.startswith("/v1/vector/"):
            return Response(404, json={"error": "not_found", "message": "collection not found"})
        return Response(500, json={"error": "internal", "message": "unexpected"})

    return MockTransport(handler)


def _client_with_transport() -> Client:
    client = Client("http://test", api_key="dev", timeout=1.0)
    client._http.close()
    client._http = httpx.Client(
        base_url="http://test",
        timeout=1.0,
        headers={"Authorization": "Bearer dev"},
        transport=_mock_transport(),
    )
    return client


class VectorAPITests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _client_with_transport()

    def tearDown(self) -> None:
        self.client.close()

    def test_list_and_info(self) -> None:
        collections = self.client.vector.list()
        self.assertTrue(collections)
        self.assertEqual(collections[0]["collection"], "docs")

        info = self.client.vector.info("docs")
        self.assertEqual(info["collection"], "docs")
        self.assertEqual(info["dim"], 3)

    def test_info_not_found_raises(self) -> None:
        with self.assertRaises(RustKissVDBError) as exc:
            self.client.vector.info("missing")
        self.assertIn("not_found", str(exc.exception))


if __name__ == "__main__":
    unittest.main()
