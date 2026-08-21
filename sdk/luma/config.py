from typing import Any, Dict, Optional

from ._http import Http


class ConfigClient:
    def __init__(self, http: Http):
        self._http = http

    def get(self) -> dict:
        return self._http.get("/v1/config")

    async def aget(self) -> dict:
        return await self._http.aget("/v1/config")

    def put(self, config: dict) -> dict:
        return self._http.put("/v1/config", config)

    async def aput(self, config: dict) -> dict:
        return await self._http.aput("/v1/config", config)

    def probe_embedding(
        self,
        provider: str,
        *,
        url: str = "",
        api_key: str = "",
        model: str = "",
        azure_api_base: str = "",
        azure_deployment: str = "",
        azure_api_version: str = "",
    ) -> dict:
        """Test an embedding configuration and measure its real dimension.

        Embeds a short probe string and reports the dimension the provider
        actually returns, which is the reliable way to set ``embedding_dim``
        instead of typing it from memory.

        Always answers HTTP 200: check ``ok``. When false, ``error`` carries the
        provider's own message, so this never raises for a provider-side
        failure — only for auth or role problems.
        """
        return self._http.post(
            "/v1/config/embedding/probe",
            _probe_body(provider, url, api_key, model,
                        azure_api_base, azure_deployment, azure_api_version),
        )

    async def aprobe_embedding(
        self,
        provider: str,
        *,
        url: str = "",
        api_key: str = "",
        model: str = "",
        azure_api_base: str = "",
        azure_deployment: str = "",
        azure_api_version: str = "",
    ) -> dict:
        return await self._http.apost(
            "/v1/config/embedding/probe",
            _probe_body(provider, url, api_key, model,
                        azure_api_base, azure_deployment, azure_api_version),
        )


def _probe_body(provider: str, url: str, api_key: str, model: str,
                azure_api_base: str, azure_deployment: str,
                azure_api_version: str) -> Dict[str, Any]:
    return {
        "provider": provider,
        "url": url,
        "api_key": api_key,
        "model": model,
        "azure_api_base": azure_api_base,
        "azure_deployment": azure_deployment,
        "azure_api_version": azure_api_version,
    }
