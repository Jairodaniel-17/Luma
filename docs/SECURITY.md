# SECURITY.md

## 1. Modos

| Modo            | Descripción                                                      |
|-----------------|------------------------------------------------------------------|
| Local (default) | `bind=127.0.0.1`, sin auth. Ideal para desarrollo.               |
| Protegido       | Bind explícito `--bind 0.0.0.0` o `--unsafe-bind` + proxy (Caddy/NGINX) con Basic Auth/mTLS. |
| API Key         | `RUSTKISS_API_KEY` (o `API_KEY`) activa `Authorization: Bearer`. Si no se exporta, las rutas siguen abiertas. |

## 2. Checklist rápido

1. **No expongas `0.0.0.0` sin un proxy**. El binario ahora exige `--bind 0.0.0.0` o `--unsafe-bind` explícito para hacerlo evidente.
2. **Envía `RUSTKISS_API_KEY`** en entornos compartidos. El middleware valida `Authorization: Bearer …` en todas las rutas HTTP (SSE incluido).
3. **Limita orígenes** si usas CORS: `CORS_ALLOWED_ORIGINS=https://miapp.example` evita wildcard.
4. **Proxy recomendado (snippet Caddy)**:

```caddyfile
rustkiss.mydomain.com {
    reverse_proxy localhost:9917
    basicauth /* {
        admin JDJhJDEwJFV3ZW9OYk5...
    }
}
```

5. **TLS**: deja el TLS al proxy (Caddy/NGINX/Traefik). El binario no incluye TLS nativo para mantenerlo KISS.

## 3. Variables útiles

| Variable              | Explicación                                         |
|-----------------------|-----------------------------------------------------|
| `BIND_ADDR`           | Override del bind por env (ej. `BIND_ADDR=0.0.0.0`). |
| `RUSTKISS_API_KEY`    | Token que se compara contra `Authorization: Bearer`. |
| `SQLITE_ENABLED`      | Sólo habilítalo si `DATA_DIR` está en una ruta privada. |
| `SQLITE_DB_PATH`      | Ruta personalizada (por defecto `DATA_DIR/sqlite/rustkiss.db`). |

## 4. SSE y tiempo real

- El stream `/v1/stream` hereda las mismas reglas de auth.
- Si usas proxies, habilita `X-Accel-Buffering: no` (NGINX) o `response buffering off` para no romper SSE.

## 5. Buenas prácticas adicionales

- **Backups**: usa snapshots (`Persist::write_snapshot`) + WAL y cópialos a storage cifrado.
- **Rotación de API key**: acepta un `Authorization` listado en un Vault externo y refresca el proceso (o implementa un endpoint admin si lo necesitas).
- **Logs**: no logeamos bodies ni API keys. Si los necesitas, usa el proxy para offloading/auditoría.
