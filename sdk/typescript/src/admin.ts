import { HttpClient } from "./http.js";
import type { AuditEntry, AuditQueryOptions, BackupResult } from "./types.js";

export class AdminClient {
  constructor(private readonly http: HttpClient) {}

  /** Trigger a snapshot and return the current WAL offset. Requires admin role. */
  backup(): Promise<BackupResult> {
    return this.http.post("/v1/admin/backup");
  }

  /** Query the audit log. Requires admin role. */
  audit(opts: AuditQueryOptions = {}): Promise<AuditEntry[]> {
    const params: Record<string, string | number | boolean | undefined> = {
      limit: opts.limit ?? 100,
    };
    if (opts.from_ms !== undefined) params["from_ms"] = opts.from_ms;
    if (opts.to_ms !== undefined) params["to_ms"] = opts.to_ms;
    if (opts.key !== undefined) params["key"] = opts.key;
    return this.http.get("/v1/admin/audit", params);
  }
}
