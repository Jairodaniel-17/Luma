from __future__ import annotations

import argparse
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from rustkissvdb import Config, RustKissVDBError


def now_ts() -> int:
    return int(time.time())


class ConversationStore:
    """
    Guarda un historial de chat en la State Store:
      key = chat:{session_id}:history
      value = {"messages": [{"role","content","ts"}...]}
    """

    def __init__(self, client, session_id: str) -> None:
        self.client = client
        self.session_id = session_id
        self.key = f"chat:{session_id}:history"
        self.ensure_initialized()

    def ensure_initialized(self) -> None:
        try:
            self.client.state.get(self.key)
        except RustKissVDBError:
            self.client.state.put(self.key, {"messages": []})

    def load(self) -> Tuple[List[Dict[str, Any]], Optional[int]]:
        try:
            item = self.client.state.get(self.key)
        except RustKissVDBError:
            return [], None
        value = item.get("value") or {}
        messages = value.get("messages", [])
        return list(messages), item.get("revision")

    def append(self, role: str, content: str) -> None:
        for _ in range(6):
            history, revision = self.load()
            history.append({"role": role, "content": content, "ts": now_ts()})
            try:
                self.client.state.put(
                    self.key,
                    {"messages": history},
                    if_revision=revision,
                )
                return
            except RustKissVDBError as exc:
                if "revision" in str(exc).lower():
                    continue
                raise
        raise RuntimeError("revision_mismatch persistente, reintentos agotados")

    def clear(self) -> None:
        self.client.state.put(self.key, {"messages": []})

    def display(self) -> None:
        history, _revision = self.load()
        if not history:
            print("Historial vacio\n")
            return
        print("Historial actual:")
        for msg in history[-50:]:
            ts = msg.get("ts")
            role = msg.get("role", "?")
            content = msg.get("content", "")
            print(f"[{ts}] {role}: {content}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejemplo: guardar historial de chat en RustKissVDB State Store",
    )
    parser.add_argument(
        "--session",
        default=os.getenv("CHAT_SESSION_ID") or uuid.uuid4().hex[:12],
        help="ID de la sesion (default: CHAT_SESSION_ID o random)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.from_env()
    with cfg.create_client() as client:
        store = ConversationStore(client, args.session)
        print(f"Sesion: {args.session}")
        print("Comandos:")
        print("  /show          -> mostrar historial")
        print("  /clear         -> borrar historial")
        print("  /assistant ... -> agregar mensaje como assistant")
        print("  /exit          -> salir")
        print("Cualquier otra entrada se guarda como mensaje del user.\n")

        while True:
            line = input("you> ").strip()
            if not line:
                continue
            if line.lower() in {"/exit", "exit", "quit"}:
                break
            if line.lower() == "/show":
                store.display()
                continue
            if line.lower() == "/clear":
                store.clear()
                print("Historial limpiado.\n")
                continue
            role = "user"
            content = line
            if line.startswith("/assistant "):
                role = "assistant"
                content = line[len("/assistant ") :].strip()
            store.append(role, content)
            print(f"Guardado ({role}).")


if __name__ == "__main__":
    main()
