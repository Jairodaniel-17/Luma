import { AdminClient } from "./admin.js";
import { AuthClient } from "./auth.js";
import { DiskAnnClient } from "./diskann.js";
import { DocClient } from "./doc.js";
import { EventsClient } from "./events.js";
import { HttpClient } from "./http.js";
import { HubClient } from "./hub.js";
import { MemoryClient } from "./memory.js";
import { StateClient } from "./state.js";
import { VectorClient } from "./vector.js";
import type { LumaClientOptions } from "./types.js";

/**
 * LumaClient — the main entry point for the Luma SDK.
 *
 * ```typescript
 * import { LumaClient } from 'luma-vdb';
 *
 * const client = new LumaClient({
 *   baseUrl: 'http://localhost:1234',
 *   apiKey: 'my-secret-key',
 * });
 * ```
 */
export class LumaClient {
  private readonly http: HttpClient;

  /** Core vector operations: collections, upsert, search, rerank, aggregate. */
  readonly vector: VectorClient;

  /** Key-value store with TTL and optimistic locking (CAS). */
  readonly state: StateClient;

  /** Raw JSON document store. */
  readonly doc: DocClient;

  /** SSE event stream from the pub/sub bus. */
  readonly events: EventsClient;

  /** API key management (admin only). */
  readonly auth: AuthClient;

  /** Admin operations: backup and audit log (admin only). */
  readonly admin: AdminClient;

  constructor(options: LumaClientOptions) {
    this.http = new HttpClient(options);
    this.vector = new VectorClient(this.http);
    this.state = new StateClient(this.http);
    this.doc = new DocClient(this.http);
    this.events = new EventsClient(this.http);
    this.auth = new AuthClient(this.http);
    this.admin = new AdminClient(this.http);
  }

  /**
   * Return a `HubClient` scoped to the given namespace.
   * The LumaDatabase hub handles auto-chunking, embedding, and hybrid search.
   */
  db(namespace: string): HubClient {
    return new HubClient(this.http, namespace);
  }

  /**
   * Return a `MemoryClient` scoped to the given namespace.
   * NS-Mem provides episodic, semantic, procedural, and working memory.
   */
  memory(namespace: string): MemoryClient {
    return new MemoryClient(this.http, namespace);
  }

  /**
   * Return a `DiskAnnClient` scoped to the given vector collection.
   * Manages the Vamana disk-based ANN graph index.
   */
  diskann(collection: string): DiskAnnClient {
    return new DiskAnnClient(this.http, collection);
  }

  /** Check server health. */
  health(): Promise<Record<string, unknown>> {
    return this.http.get("/v1/health");
  }

  /** Retrieve Prometheus-compatible metrics text. */
  metrics(): Promise<string> {
    return this.http.get("/v1/metrics");
  }
}
