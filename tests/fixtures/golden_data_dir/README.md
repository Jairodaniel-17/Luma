# Golden `data_dir` fixture

A real `data_dir` written by a build of this project. `tests/golden_data_dir.rs`
starts the **current** binary over a copy of it and reads every record back.

It exists to enforce rule 1 of the data compatibility policy in
`docs/SPEC-producto.md`: every version reads what the previous one wrote. Rule 4
says CI must check it, because a rule nothing checks is a wish.

## What is in here

| Path | Format under test |
|---|---|
| `events-NNNNNN.log` | segmented WAL, JSON lines with per-record checksums |
| `snapshot.json` | state snapshot |
| `state.redb` | redb-backed KV projection |
| `vectors/golden/manifest.json` | collection manifest — written **before** the embedding-provenance fields existed, which is the point |
| `vectors/golden/runs/*.log` | vector run records |
| `vectors/golden/*.mmap` | mmap-backed vector and q8 storage |
| `blobs/golden/obj.bin` | object storage layout |
| `queues/golden/*.json` | queue message format |

Every record carries a `marker` naming the version that wrote it, so a failure
says which vintage stopped being readable.

## Rules

- **Never edit it by hand.** It is a recording, not a config.
- **Never regenerate it to make the test pass.** A failure means either the
  format change needs a migration, or it does not belong in v1.
- **Regenerate on release**, from that release's binary, and commit alongside.
  Keeping older vintages in sibling directories is fine and cheap: the whole
  tree compresses to a few KB.
