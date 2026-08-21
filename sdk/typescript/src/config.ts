import { HttpClient } from "./http.js";
import type { EmbeddingProbeRequest, EmbeddingProbeResult } from "./types.js";

/**
 * Runtime configuration (`/v1/config`). Admin only.
 *
 * Embedding settings take effect immediately after `update`; every other field
 * still needs a server restart. The response says which is which.
 */
export class ConfigClient {
  constructor(private readonly http: HttpClient) {}

  /** Read the instance configuration. Secrets are never serialized. */
  get(): Promise<Record<string, unknown>> {
    return this.http.get("/v1/config");
  }

  /** Persist configuration to `luma.toml` and hot-reload the embedding client. */
  update(config: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.http.put("/v1/config", config);
  }

  /**
   * Test an embedding configuration and measure its real output dimension.
   *
   * Embeds a short probe string and reports the dimension the provider actually
   * returns — the reliable way to set `embedding_dim` instead of typing it from
   * memory.
   *
   * Always answers HTTP 200, so check `ok`: when false, `error` carries the
   * provider's own message. This never throws for a provider-side failure, only
   * for auth or role problems.
   */
  probeEmbedding(request: EmbeddingProbeRequest): Promise<EmbeddingProbeResult> {
    return this.http.post("/v1/config/embedding/probe", {
      provider: request.provider,
      url: request.url ?? "",
      api_key: request.api_key ?? "",
      model: request.model ?? "",
      azure_api_base: request.azure_api_base ?? "",
      azure_deployment: request.azure_deployment ?? "",
      azure_api_version: request.azure_api_version ?? "",
    });
  }
}
