# Demo

1. Levanta el servidor (puedes fijar el nivel con `--logs info|warning|error|critical`; default `info`):

```bash
set DATA_DIR=.\data
cargo run --bin rust-kiss-vdb -- --logs info
```

2. Si quieres ver la especificación, abre <http://localhost:8080/docs> o descarga <http://localhost:8080/openapi.yaml>.

3. En otra terminal, abre SSE:

```bash
curl -N "http://localhost:8080/v1/stream?since=0&types=state_updated,state_deleted,vector_added,vector_upserted,vector_updated,vector_deleted,gap" ^
```

4. Corre el script:

- PowerShell: `scripts/demo.ps1`

> No es necesario enviar headers de autenticación en esta versión. Ajusta los logs con `--logs warning` si quieres menos ruido.
