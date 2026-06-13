import { HttpClient } from "./http.js";
import type { DocFindOptions, DocFindResult, Metadata } from "./types.js";

export class DocClient {
  constructor(private readonly http: HttpClient) {}

  put(
    collection: string,
    id: string,
    document: Metadata,
  ): Promise<Record<string, unknown>> {
    return this.http.put(
      `/v1/doc/${encodeURIComponent(collection)}/${encodeURIComponent(id)}`,
      document,
    );
  }

  get(collection: string, id: string): Promise<Metadata> {
    return this.http.get(
      `/v1/doc/${encodeURIComponent(collection)}/${encodeURIComponent(id)}`,
    );
  }

  delete(collection: string, id: string): Promise<void> {
    return this.http.delete(
      `/v1/doc/${encodeURIComponent(collection)}/${encodeURIComponent(id)}`,
    );
  }

  find(
    collection: string,
    opts: DocFindOptions = {},
  ): Promise<DocFindResult> {
    return this.http.post(
      `/v1/doc/${encodeURIComponent(collection)}/find`,
      { filter: opts.filter ?? null, limit: opts.limit ?? 20 },
    );
  }
}
