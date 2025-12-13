# RustKissVDB

DB KISS en Rust: **State Store + Event Store + SSE + Vector Store**.

## Quickstart

1. Levanta el servidor:

   ```bash
   set DATA_DIR=.\data
   cargo run --bin rust-kiss-vdb
   ```

2. Abre un stream SSE:

   ```bash
   curl -N "http://localhost:8080/v1/stream?since=0"
   ```

3. Ejecuta el demo para escribir estado/vectores:
   ```powershell
   scripts\demo.ps1
   ```

> Nota: la v1 expone todos los endpoints sin autenticación; añade tu propio proxy si necesitas protegerlos.

Documentación: `docs/` (arquitectura, API, config, demo, bench, OpenAPI).
