import { HttpClient } from "./http.js";
import type { DiskAnnBuildOptions, DiskAnnStatus } from "./types.js";

/**
 * DiskAnnClient — manage the DiskANN (Vamana graph) index for a collection.
 *
 * Obtain via `client.diskann(collection)`.
 */
export class DiskAnnClient {
  private readonly base: string;

  constructor(
    private readonly http: HttpClient,
    collection: string,
  ) {
    this.base = `/v1/vector/${encodeURIComponent(collection)}/diskann`;
  }

  /** Build the DiskANN graph index for the collection. */
  build(opts: DiskAnnBuildOptions = {}): Promise<Record<string, unknown>> {
    return this.http.post(`${this.base}/build`, buildBody(opts));
  }

  /** Tune DiskANN search parameters without rebuilding the graph. */
  tune(opts: DiskAnnBuildOptions = {}): Promise<Record<string, unknown>> {
    return this.http.post(`${this.base}/tune`, buildBody(opts));
  }

  /** Return the current DiskANN index status. */
  status(): Promise<DiskAnnStatus> {
    return this.http.get(`${this.base}/status`);
  }
}

function buildBody(opts: DiskAnnBuildOptions): Record<string, unknown> {
  const body: Record<string, unknown> = {};
  if (opts.max_degree !== undefined) body["max_degree"] = opts.max_degree;
  if (opts.build_threads !== undefined) body["build_threads"] = opts.build_threads;
  if (opts.search_list_size !== undefined) body["search_list_size"] = opts.search_list_size;
  return body;
}
