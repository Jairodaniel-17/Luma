import { HttpClient } from "./http.js";
import type {
  BatchPutOperation,
  JsonValue,
  StateEntry,
  StatePutOptions,
} from "./types.js";

export class StateClient {
  constructor(private readonly http: HttpClient) {}

  list(
    prefix?: string,
    limit = 100,
  ): Promise<StateEntry[]> {
    const params: Record<string, string | number | boolean | undefined> = { limit };
    if (prefix !== undefined) params["prefix"] = prefix;
    return this.http.get("/v1/state", params);
  }

  get(key: string): Promise<StateEntry> {
    return this.http.get(`/v1/state/${encodeKey(key)}`);
  }

  put(
    key: string,
    value: JsonValue,
    opts: StatePutOptions = {},
  ): Promise<StateEntry> {
    const body: Record<string, unknown> = { value };
    if (opts.ttl_ms !== undefined) body["ttl_ms"] = opts.ttl_ms;
    if (opts.if_revision !== undefined) body["if_revision"] = opts.if_revision;
    return this.http.put(`/v1/state/${encodeKey(key)}`, body);
  }

  delete(key: string): Promise<void> {
    return this.http.delete(`/v1/state/${encodeKey(key)}`);
  }

  batchPut(
    operations: BatchPutOperation[],
  ): Promise<Record<string, unknown>> {
    return this.http.post("/v1/state/batch_put", { operations });
  }

  createIndex(field: string): Promise<Record<string, unknown>> {
    return this.http.post("/v1/state/indexes", { field });
  }

  queryIndex(
    field: string,
    value: string,
    limit = 100,
  ): Promise<StateEntry[]> {
    return this.http.get(`/v1/state/index/${encodeURIComponent(field)}/${encodeURIComponent(value)}`, { limit });
  }
}

function encodeKey(key: string): string {
  // Keys may contain slashes intentionally; only encode unsafe chars.
  return key.split("/").map(encodeURIComponent).join("/");
}
