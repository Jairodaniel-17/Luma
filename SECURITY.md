# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| < 0.2   | :x:                |

## Reporting a Vulnerability

Please report vulnerabilities to the maintainers via GitHub Issues (labeled 'security') or email if available.

## Threat Model (STRIDE)

| Threat | Description | Mitigation | Status |
| :--- | :--- | :--- | :--- |
| **Spoofing** | User impersonating another user. | API Key Auth (Bearer). | Implemented |
| **Tampering** | Modifying vectors/data on disk or transit. | File permissions, TLS (reverse proxy rec). | Operationally Managed |
| **Repudiation** | User denies performing an action. | Access Logs (TraceLayer). | Implemented |
| **Information Disclosure** | Leaking sensitive vectors or metadata. | Auth checks, Error masking (no stack traces in prod). | Partially Implemented |
| **Denial of Service** | Exhausting resources (CPU/RAM/Disk). | Body limits, Vector dim limits, timeouts. | Hardened in Phase D |
| **Elevation of Privilege** | Gaining admin access. | API Key scopes (future work). Currently single key. | Accepted Risk |

## Specific Risks & Mitigations

### 1. SQL Injection (rusqlite)
- **Risk**: Attackers executing arbitrary SQL via `sql/query` or `sql/exec`.
- **Mitigation**: 
  - `rusqlite` uses parameterized queries (`?`). 
  - The API allows arbitrary SQL by design for authorized users, but the internal code must strictly use parameters.
  - Hardening: Ensure all internal queries use parameters.

### 2. DoS via Large Payloads
- **Risk**: Sending massive JSON bodies or vectors to crash the server.
- **Mitigation**:
  - `axum::extract::DefaultBodyLimit` configured via `MAX_BODY_BYTES`.
  - Application-level checks for `max_vector_dim`, `max_json_bytes`.
  - `tokio::time::timeout` on requests.

### 3. Path Traversal
- **Risk**: Accessing files outside `DATA_DIR`.
- **Mitigation**:
  - `src/server.rs` uses `fs::canonicalize` on startup.
  - Internal engines use relative paths joined to `DATA_DIR`.
  - No user-supplied file paths are accepted (only collection names which are validated for length/chars).

### 4. Input Validation
- **Risk**: Buffer overflows or logic errors via malformed input.
- **Mitigation**:
  - Strict typing with `serde`.
  - Explicit length checks for IDs, Keys, Collections, Vectors.
  - `validator` crate integration (future).

## Operational Security Recommendations

1. **Reverse Proxy**: Always run Luma behind Nginx/Caddy for TLS termination and additional rate limiting.
2. **Firewall**: Restrict access to the Luma port (default 8080) to trusted internal networks.
3. **Secrets**: Rotate `LUMA_API_KEY` regularly. Use environment variables, not flags, for secrets.
