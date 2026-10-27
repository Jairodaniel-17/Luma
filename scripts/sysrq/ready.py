#!/usr/bin/env python3
"""Wait until Luma answers /v1/health/ready with 200 (up to 300 s). Runs in the VM."""
import time
import urllib.request

for _ in range(1500):
    try:
        if urllib.request.urlopen("http://127.0.0.1:7700/v1/health/ready", timeout=1).status == 200:
            raise SystemExit(0)
    except SystemExit:
        raise
    except Exception:
        time.sleep(0.2)
raise SystemExit("Luma did not become ready within 300 s")
