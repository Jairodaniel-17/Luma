import { HttpClient } from "./http.js";
import type {
  HubIngestOptions,
  HubSearchOptions,
  HubSearchResult,
  Metadata,
} from "./types.js";

/**
 * HubClient — LumaDatabase hybrid layer for a single namespace.
 * Handles auto-chunking, embedding generation, and hybrid SQL+vector search.
 *
 * Obtain via `client.db(namespace)`.
 */
export class HubClient {
  private readonly base: string;

  constructor(
    private readonly http: HttpClient,
    namespace: string,
  ) {
    this.base = `/v1/db/${encodeURIComponent(namespace)}`;
  }

  /**
   * Ingest a text document. The server chunks it, generates embeddings, and
   * stores both the vector and the raw text.
   */
  ingest(
    text: string,
    opts: HubIngestOptions = {},
  ): Promise<Record<string, unknown>> {
    const body: Record<string, unknown> = { text };
    if (opts.id !== undefined) body["id"] = opts.id;
    if (opts.metadata !== undefined) body["metadata"] = opts.metadata;
    return this.http.post(`${this.base}/ingest`, body);
  }

  /**
   * Hybrid search: SQL pre-filter followed by vector similarity ranking.
   */
  search(
    query: string,
    opts: HubSearchOptions = {},
  ): Promise<HubSearchResult> {
    const body: Record<string, unknown> = { query, limit: opts.limit ?? 10 };
    if (opts.sql_filter !== undefined) body["sql_filter"] = opts.sql_filter;
    return this.http.post(`${this.base}/search`, body);
  }
}
