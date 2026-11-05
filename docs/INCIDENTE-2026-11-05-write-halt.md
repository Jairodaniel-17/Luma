# Incidente: Luma colgado por «write halt because of write buffer saturation» (05/11/2026)

## Sintoma
En celestia, Luma dejo de responder (`/v1/health` sin respuesta, puerto 8093 sin escuchar tras reiniciar) y el journal de
systemd se lleno con cientos de miles de lineas `fjall::partition: partition: write halt because of write buffer saturation`
(unas 100 por segundo). Dejaron de servir las portadas de Musica (cache en Luma) y la cache de IA de Friochap.
Reiniciar no lo arreglaba: volvia al mismo bucle antes de abrir el puerto.

## Causa (probable, NO reproducida)
`StateDb::open` usaba la configuracion por defecto de fjall (buffer de escritura de **64 MiB**) mientras Luma admite documentos de
hasta 100 MB (`max_json_mb`) y lotes arbitrarios. En fjall 2.11, `check_write_buffer_size` espera en bucle (10 ms) mientras el
buffer total este lleno; si el flush no avanza, toda escritura queda esperando, y al reabrir el diario (146 MB) se reproduce en
memtables y puede ocurrir lo mismo antes de que el servidor escuche.

Lo que se probo (05/11/2026, `cargo test --lib write_halt_tests`): un valor de 70 MB seguido de escrituras pequenas y reapertura, y
300 MiB en valores de 1 MiB sin persistir y reapertura. **Las dos pasan tambien con el buffer de 64 MiB**, asi que el disparador exacto
de celestia (a las 04:09 UTC, tras la carga que hizo musica-covers y las subidas de demos) no se reprodujo con una prueba de una sola
hebra. Quedan como hipotesis un flush detenido por E/S lenta o una carrera con varias hebras concurrentes. Las mitigaciones de abajo
son validas aunque la causa exacta siga abierta, y las pruebas quedan como regresion.

## Receta de recuperacion (probada en celestia)
1. `systemctl stop luma`
2. Respaldar: `tar czf /root/respaldos-luma/luma-$(date +%F-%H%M).tgz -C /var/lib/private luma`
3. Mover el diario aparte: `mv /var/lib/private/luma/state.lsm/journals /root/respaldos-luma/journals-viejos` y
   `mkdir` uno vacio con el mismo dueno (`chown 64912:64912` en celestia). Luma reconstruye el estado desde su propio WAL de eventos
   (`events-*.log`, `wal_retention_segments`).
4. `systemctl start luma` y comprobar `GET /v1/health` y que no haya lineas `write halt`.

## Arreglo en el codigo
- Buffer de escritura y diario dimensionados desde `max_json_mb` (nunca menores que el mayor documento admitido, por varias
  particiones) y configurables con `LUMA_LSM_WRITE_BUFFER_MB`.
- El filtro de logs por defecto silencia `fjall` por debajo de `warn`, para que un bucle no llene el journal.
- Pruebas de regresion `un_valor_grande_no_deja_las_escrituras_en_espera_para_siempre` y `carga_sostenida_y_reapertura_no_se_cuelgan`
  (`src/engine/state_db.rs`).
- Pendiente: un vigilante que, si una escritura espera al buffer mas de N s, devuelva 503 en vez de colgar el hilo (requiere
  `spawn_blocking` con plazo en la ruta de escritura; no se hizo porque no se reprodujo el bloqueo).
