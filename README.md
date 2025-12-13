# RustKissVDB

DB KISS en Rust: **State Store + Event Store + SSE + Vector Store**.

## Quickstart

1. Levanta el servidor (opcionalmente, ajusta el nivel con `--logs {info|warning|error|critical}`; default `info`):

   ```bash
   set DATA_DIR=.\data
   cargo run --bin rust-kiss-vdb -- --logs info
   ```

2. Abre un stream SSE:

   ```bash
   curl -N "http://localhost:8080/v1/stream?since=0"
   ```

3. Explora la documentación en vivo:
   - Docs: <http://localhost:8080/docs>
   - Esquema: <http://localhost:8080/openapi.yaml>

4. Ejecuta el demo para escribir estado/vectores:
   ```powershell
   scripts\demo.ps1
   ```

> Notas: la v1 expone todos los endpoints sin autenticación y puedes controlar el ruido de logs con `--logs` (ej. `--logs warning`).

Documentación: `docs/` (arquitectura, API, config, demo, bench, OpenAPI).
