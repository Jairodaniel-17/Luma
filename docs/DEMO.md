# Demo

1. Levanta el servidor:

```bash
set DATA_DIR=.\data
cargo run --bin rust-kiss-vdb
```

2. En otra terminal, abre SSE:

```bash
curl -N "http://localhost:8080/v1/stream?since=0&types=state_updated,state_deleted,vector_added,vector_upserted,vector_updated,vector_deleted,gap" ^
```

3. Corre el script:

- PowerShell: `scripts/demo.ps1`

> No es necesario enviar headers de autenticación en esta versión.
