import { HttpClient } from "./http.js";
import type {
  BeliefVersion,
  CentralityResult,
  CreateEdgeRequest,
  IngestEventOptions,
  MemoryEdge,
  MemoryQueryOptions,
  MemoryQueryResult,
  MemoryRecord,
  NextStepOptions,
  NodeEdgesResult,
  Procedure,
  ProcedureStep,
  UpsertFactOptions,
} from "./types.js";

/**
 * MemoryClient — NS-Mem agent memory layer for a single namespace.
 * Covers episodic, semantic, procedural, and working memory.
 *
 * Obtain via `client.memory(namespace)`.
 */
export class MemoryClient {
  private readonly ns: string;

  constructor(
    private readonly http: HttpClient,
    namespace: string,
  ) {
    this.ns = `/v1/memory/${encodeURIComponent(namespace)}`;
  }

  // ── Episodic events ──────────────────────────────────────────────────────

  /**
   * Store an episodic event. Triggers the LLM consolidation pipeline when
   * enabled (extracts semantic facts from the event text).
   */
  ingestEvent(
    text: string,
    opts: IngestEventOptions = {},
  ): Promise<Record<string, unknown>> {
    const body: Record<string, unknown> = { text };
    setOpt(body, "id", opts.id);
    setOpt(body, "entity_id", opts.entity_id);
    setOpt(body, "metadata", opts.metadata);
    setOpt(body, "source", opts.source);
    setOpt(body, "session_id", opts.session_id);
    setOpt(body, "expires_at_ms", opts.expires_at_ms);
    return this.http.post(`${this.ns}/ingest_event`, body);
  }

  // ── Semantic facts ───────────────────────────────────────────────────────

  /**
   * Create or update a semantic fact. Versions the previous value and creates
   * a Supersedes (or Contradicts) edge on overwrite.
   */
  upsertFact(
    content: string,
    opts: UpsertFactOptions = {},
  ): Promise<Record<string, unknown>> {
    const body: Record<string, unknown> = { content };
    setOpt(body, "id", opts.id);
    setOpt(body, "entity_id", opts.entity_id);
    setOpt(body, "fact_key", opts.fact_key);
    setOpt(body, "metadata", opts.metadata);
    setOpt(body, "source", opts.source);
    setOpt(body, "confidence", opts.confidence);
    setOpt(body, "status", opts.status);
    return this.http.post(`${this.ns}/upsert_fact`, body);
  }

  // ── Procedural memory ────────────────────────────────────────────────────

  /**
   * Register or replace a procedural DAG (nodes + typed edges + constraints).
   */
  upsertProcedure(proc: Procedure): Promise<Record<string, unknown>> {
    const body: Record<string, unknown> = {
      procedure_id: proc.procedure_id,
      name: proc.name,
      nodes: proc.nodes,
      edges: proc.edges,
    };
    setOpt(body, "version", proc.version);
    setOpt(body, "status", proc.status);
    setOpt(body, "description", proc.description);
    setOpt(body, "confidence", proc.confidence);
    setOpt(body, "source", proc.source);
    setOpt(body, "constraints", proc.constraints);
    return this.http.post(`${this.ns}/upsert_procedure`, body);
  }

  // ── Query ────────────────────────────────────────────────────────────────

  /**
   * Query across episodic, semantic, and procedural memory layers. Mode is
   * auto-detected from the query text when not specified.
   */
  query(
    query: string,
    opts: MemoryQueryOptions = {},
  ): Promise<MemoryQueryResult> {
    const body: Record<string, unknown> = { query };
    setOpt(body, "entity_id", opts.entity_id);
    setOpt(body, "session_id", opts.session_id);
    setOpt(body, "procedure_id", opts.procedure_id);
    setOpt(body, "current_node_id", opts.current_node_id);
    setOpt(body, "context", opts.context);
    setOpt(body, "mode", opts.mode);
    setOpt(body, "limit", opts.limit);
    setOpt(body, "include_evidence", opts.include_evidence);
    setOpt(body, "include_plan", opts.include_plan);
    setOpt(body, "include_diagnostics", opts.include_diagnostics);
    return this.http.post(`${this.ns}/query`, body);
  }

  // ── Next step ────────────────────────────────────────────────────────────

  /**
   * Return the next valid node in a procedure given the current node and
   * evaluation context.
   */
  nextStep(
    procedureId: string,
    opts: NextStepOptions = {},
  ): Promise<ProcedureStep> {
    const body: Record<string, unknown> = { procedure_id: procedureId };
    setOpt(body, "current_node_id", opts.current_node_id);
    setOpt(body, "context", opts.context);
    return this.http.post(`${this.ns}/next_step`, body);
  }

  // ── Timeline ─────────────────────────────────────────────────────────────

  /** Fetch the episodic timeline for an entity, ordered chronologically. */
  timeline(entityId: string, limit?: number): Promise<MemoryRecord[]> {
    const params: Record<string, string | number | boolean | undefined> = {};
    if (limit !== undefined) params["limit"] = limit;
    return this.http.get(
      `${this.ns}/timeline/${encodeURIComponent(entityId)}`,
      params,
    );
  }

  // ── Graph edges ──────────────────────────────────────────────────────────

  /** Create a typed weighted edge between two memory records. */
  createEdge(req: CreateEdgeRequest): Promise<Record<string, unknown>> {
    const body: Record<string, unknown> = {
      source_id: req.source_id,
      target_id: req.target_id,
      edge_type: req.edge_type,
    };
    setOpt(body, "id", req.id);
    setOpt(body, "weight", req.weight);
    setOpt(body, "metadata", req.metadata);
    return this.http.post(`${this.ns}/edges`, body);
  }

  /** Return all outgoing and incoming edges for a memory node. */
  nodeEdges(memoryId: string): Promise<NodeEdgesResult> {
    return this.http.get(
      `${this.ns}/edges/${encodeURIComponent(memoryId)}`,
    );
  }

  /** Delete a memory edge by its ID. */
  deleteEdge(edgeId: string): Promise<Record<string, unknown>> {
    return this.http.post(
      `${this.ns}/edges/${encodeURIComponent(edgeId)}/delete`,
    );
  }

  // ── Belief versioning ────────────────────────────────────────────────────

  /** Return the full version history of a semantic fact by its fact_key. */
  beliefHistory(factKey: string): Promise<BeliefVersion[]> {
    return this.http.get(
      `${this.ns}/beliefs/${encodeURIComponent(factKey)}/history`,
    );
  }

  // ── Centrality ───────────────────────────────────────────────────────────

  /** Recompute PageRank centrality scores for all nodes in the namespace. */
  refreshCentrality(): Promise<CentralityResult> {
    return this.http.post(`${this.ns}/graph/centrality`);
  }
}

function setOpt(
  body: Record<string, unknown>,
  key: string,
  value: unknown,
): void {
  if (value !== undefined && value !== null) {
    body[key] = value;
  }
}
