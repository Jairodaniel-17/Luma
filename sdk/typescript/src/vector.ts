import { HttpClient } from "./http.js";
import type {
  AggregateRequest,
  AggregateResult,
  BatchSearchResult,
  CollectionDetail,
  CollectionInfo,
  MetadataFilter,
  RerankRequest,
  RerankResult,
  ScrollOptions,
  ScrollResult,
  SearchQuery,
  SearchResult,
  VectorItem,
  VectorUpdateRequest,
} from "./types.js";

export class VectorClient {
  constructor(private readonly http: HttpClient) {}

  // ── Collection management ────────────────────────────────────────────────

  list(): Promise<{ collections: CollectionInfo[] }> {
    return this.http.get("/v1/vector");
  }

  getCollection(collection: string): Promise<CollectionDetail> {
    return this.http.get(`/v1/vector/${collection}`);
  }

  createCollection(
    collection: string,
    dim: number,
    metric: "cosine" | "euclidean" | "dot" = "cosine",
  ): Promise<Record<string, unknown>> {
    return this.http.post(`/v1/vector/${collection}`, { dim, metric });
  }

  deleteCollection(collection: string): Promise<void> {
    return this.http.delete(`/v1/vector/${collection}`);
  }

  // ── Write operations ─────────────────────────────────────────────────────

  add(
    collection: string,
    id: string,
    vector: number[],
    meta?: Record<string, unknown>,
  ): Promise<Record<string, unknown>> {
    return this.http.post(`/v1/vector/${collection}/add`, buildVecBody(id, vector, meta));
  }

  upsert(
    collection: string,
    id: string,
    vector: number[],
    meta?: Record<string, unknown>,
  ): Promise<Record<string, unknown>> {
    return this.http.post(`/v1/vector/${collection}/upsert`, buildVecBody(id, vector, meta));
  }

  upsertBatch(
    collection: string,
    items: VectorItem[],
  ): Promise<Record<string, unknown>> {
    return this.http.post(`/v1/vector/${collection}/upsert_batch`, { items });
  }

  update(
    collection: string,
    req: VectorUpdateRequest,
  ): Promise<Record<string, unknown>> {
    return this.http.post(`/v1/vector/${collection}/update`, req);
  }

  deleteItem(
    collection: string,
    id: string,
  ): Promise<Record<string, unknown>> {
    return this.http.post(`/v1/vector/${collection}/delete`, { id });
  }

  deleteBatch(
    collection: string,
    ids: string[],
  ): Promise<Record<string, unknown>> {
    return this.http.post(`/v1/vector/${collection}/delete_batch`, { ids });
  }

  // ── Read operations ──────────────────────────────────────────────────────

  getById(collection: string, id: string): Promise<VectorItem> {
    return this.http.get(`/v1/vector/${collection}/get`, { id });
  }

  // ── Search ───────────────────────────────────────────────────────────────

  search(collection: string, query: SearchQuery): Promise<SearchResult> {
    return this.http.post(`/v1/vector/${collection}/search`, query);
  }

  searchBatch(
    collection: string,
    queries: SearchQuery[],
  ): Promise<BatchSearchResult> {
    return this.http.post(`/v1/vector/${collection}/search_batch`, { queries });
  }

  // ── Scroll ───────────────────────────────────────────────────────────────

  scroll(collection: string, opts: ScrollOptions = {}): Promise<ScrollResult> {
    const params: Record<string, string | number | boolean | undefined> = {
      limit: opts.limit ?? 100,
      include_vectors: opts.include_vectors ?? false,
      cursor: opts.cursor,
    };
    return this.http.get(`/v1/vector/${collection}/scroll`, params);
  }

  // ── Rerank ───────────────────────────────────────────────────────────────

  rerank(collection: string, req: RerankRequest): Promise<RerankResult> {
    return this.http.post(`/v1/vector/${collection}/rerank`, req);
  }

  // ── Aggregate ────────────────────────────────────────────────────────────

  aggregate(
    collection: string,
    req: AggregateRequest,
  ): Promise<AggregateResult> {
    return this.http.post(`/v1/vector/${collection}/aggregate`, req);
  }
}

function buildVecBody(
  id: string,
  vector: number[],
  meta?: Record<string, unknown>,
): Record<string, unknown> {
  const body: Record<string, unknown> = { id, vector };
  if (meta !== undefined) body["meta"] = meta;
  return body;
}

// Legacy filter helper — converts simple `{field: value}` maps to typed filter
export function fromLegacyFilter(
  filters: Record<string, unknown>,
): MetadataFilter {
  const result: MetadataFilter = {};
  for (const [k, v] of Object.entries(filters)) {
    result[k] = { eq: v as import("./types.js").JsonValue };
  }
  return result;
}
