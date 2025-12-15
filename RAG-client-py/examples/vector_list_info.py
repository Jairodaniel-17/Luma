from __future__ import annotations

import json
import os
from typing import List, Dict, Any

from rustkissvdb import Config


def pretty(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def main() -> None:
    cfg = Config.from_env()
    with cfg.create_client() as client:
        collections: List[Dict[str, Any]] = client.vector.list()
        if not collections:
            print("❌ No hay colecciones vectoriales registradas en RustKissVDB.")
            return

        print("📋 Colecciones disponibles:")
        for col in collections:
            print(
                f" - {col.get('collection')} "
                f"(dim={col.get('dim')}, metric={col.get('metric')}, count={col.get('live_count')})"
            )

        preferred = os.getenv("VDB_COLLECTION") or collections[0].get("collection")
        if not preferred:
            print("⚠️ Variable VDB_COLLECTION no está definida; usando la primera colección.")
            preferred = collections[0]["collection"]

        print(f"\n🔍 Info detallada de '{preferred}':")
        info = client.vector.info(preferred)
        print(pretty(info))


if __name__ == "__main__":
    main()
