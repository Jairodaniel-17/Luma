from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .client import Client


@dataclass(frozen=True, slots=True)
class Config:
    base_url: str
    api_key: Optional[str] = None
    timeout: float = 30.0

    @classmethod
    def from_env(cls, env_path: Optional[str] = None) -> "Config":
        """
        Carga configuración desde .env (si existe) + variables de entorno.

        Variables soportadas:
          - RUSTKISS_URL o VDB_BASE_URL
          - RUSTKISS_API_KEY o API_KEY
          - RUSTKISS_TIMEOUT (segundos)
          - PORT_RUST_KISS_VDB (si no se setea base_url)
        """
        # Import local para que el SDK no dependa de python-dotenv si no lo instalaste
        try:
            from dotenv import find_dotenv, load_dotenv  # type: ignore
        except Exception:
            find_dotenv = None
            load_dotenv = None

        if load_dotenv is not None:
            if env_path == "":
                load_dotenv()  # carga env actual (sin buscar)
            else:
                # busca .env si no se pasa ruta
                path = env_path or (find_dotenv() if find_dotenv is not None else None)
                load_dotenv(path)

        base = (
            os.getenv("RUSTKISS_URL")
            or os.getenv("VDB_BASE_URL")
            or f"http://127.0.0.1:{os.getenv('PORT_RUST_KISS_VDB', '9917')}"
        ).rstrip("/")

        api_key = os.getenv("RUSTKISS_API_KEY") or os.getenv("API_KEY")
        timeout = float(os.getenv("RUSTKISS_TIMEOUT", "30"))

        return cls(base_url=base, api_key=api_key, timeout=timeout)

    def create_client(self) -> Client:
        from .client import Client
        return Client(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )
