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

## 6. Inventario de `unsafe`

W5.3 del plan maestro. **16 sitios, 4 ficheros, un solo módulo.**

Todos los demás módulos del crate llevan `#[forbid(unsafe_code)]` en
`src/lib.rs`, así que un bloque `unsafe` fuera de `vector` es un **error de
compilación**, no un comentario de revisión que alguien pueda pasar por alto.
`tests/unsafe_inventory.rs` tapa el único hueco que eso deja: un `pub mod` nuevo
sin el atributo compilaría perfectamente, y la protección dejaría de cubrir en
silencio el código más reciente, que es justo donde más se quiere.

| Fichero | Sitios | Qué es | Qué lo hace correcto |
|---|---|---|---|
| `src/vector/mmap.rs` | 5 | `unsafe impl Pod`/`Zeroable` para `MmapHeader`, y 3 `MmapMut::map_mut` | El header es `#[repr(C)]` de enteros sin padding ni punteros, que es exactamente lo que `Pod` requiere. El `map_mut` es inseguro porque otro proceso puede truncar el fichero bajo el mapeo; el fichero vive dentro del `data_dir` de Luma y solo lo escribe Luma |
| `src/vector/q8mmap.rs` | 5 | Lo mismo para `Q8Header` y su mapeo | Idéntico razonamiento |
| `src/vector/simd.rs` | 4 | `dot_avx2` y `accumulate_avx2`, más sus dos llamadas | Las funciones son `#[target_feature(enable = "avx2")]`, y llamarlas sin AVX2 es UB. Cada llamada está tras `is_x86_feature_detected!("avx2")` **y** un `len() >= 8`, porque la versión vectorizada lee de ocho en ocho |
| `src/vector/q8.rs` | 2 | `dot_i8_avx2` y su llamada | Igual, con `len() >= 32` porque procesa 32 enteros de 8 bits por iteración |

Las dos formas de que esto se rompa, y por qué no se rompen:

1. **Llamar a una función `target_feature` en una CPU que no la tiene.** Es UB,
   no un fallo. La detección es en tiempo de ejecución (`is_x86_feature_detected!`)
   y no en tiempo de compilación, así que un binario construido en una máquina con
   AVX2 y ejecutado en una sin él cae al camino escalar en vez de morir.
2. **Leer fuera del final del slice.** Las rutas AVX2 avanzan en bloques, así que
   una entrada más corta que un bloque leería memoria ajena. De ahí las
   comprobaciones de longitud junto a la de CPU: son parte de la precondición, no
   una optimización.

El camino escalar existe siempre y es el que se ejecuta cuando `simd_enabled` está
apagado, lo que da una forma de comparar resultados sin recompilar.
