import { HttpClient } from "./http.js";
import type { ApiKeyInfo, CreateKeyResult, KeyRole } from "./types.js";

export class AuthClient {
  constructor(private readonly http: HttpClient) {}

  /** List all API keys. Requires admin role. */
  listKeys(): Promise<ApiKeyInfo[]> {
    return this.http.get("/v1/auth/keys");
  }

  /** Create a new API key. Requires admin role. */
  createKey(name: string, role: KeyRole = "user"): Promise<CreateKeyResult> {
    return this.http.post("/v1/auth/keys", { name, role });
  }

  /** Revoke an API key by ID. Requires admin role. */
  revokeKey(id: string): Promise<void> {
    return this.http.delete(`/v1/auth/keys/${encodeURIComponent(id)}`);
  }
}
