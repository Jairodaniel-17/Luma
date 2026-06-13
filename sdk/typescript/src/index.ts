// ─── Main client ──────────────────────────────────────────────────────────────
export { LumaClient } from "./client.js";

// ─── Sub-clients (tree-shakeable named exports) ───────────────────────────────
export { AdminClient } from "./admin.js";
export { AuthClient } from "./auth.js";
export { DiskAnnClient } from "./diskann.js";
export { DocClient } from "./doc.js";
export { EventsClient } from "./events.js";
export { HubClient } from "./hub.js";
export { MemoryClient } from "./memory.js";
export { SqlClient } from "./sql.js";
export { StateClient } from "./state.js";
export { VectorClient } from "./vector.js";
export { fromLegacyFilter } from "./vector.js";

// ─── HTTP internals (for advanced / testing use) ──────────────────────────────
export { HttpClient } from "./http.js";

// ─── Errors ───────────────────────────────────────────────────────────────────
export {
  LumaAuthError,
  LumaConflictError,
  LumaError,
  LumaForbiddenError,
  LumaNotFoundError,
} from "./errors.js";

// ─── Types ────────────────────────────────────────────────────────────────────
export type {
  AggregateRequest,
  AggregateResult,
  ApiKeyInfo,
  AuditEntry,
  AuditQueryOptions,
  BackupResult,
  BatchPutOperation,
  BatchSearchResult,
  BeliefVersion,
  CentralityResult,
  CollectionDetail,
  CollectionInfo,
  CreateEdgeRequest,
  CreateKeyResult,
  DiskAnnBuildOptions,
  DiskAnnStatus,
  DocFindOptions,
  DocFindResult,
  EdgeType,
  EventStreamOptions,
  FilterOp,
  HubIngestOptions,
  HubSearchHit,
  HubSearchOptions,
  HubSearchResult,
  IngestEventOptions,
  JsonValue,
  KeyRole,
  LumaClientOptions,
  LumaEvent,
  MemoryEdge,
  MemoryQueryOptions,
  MemoryQueryResult,
  MemoryRecord,
  MemoryStatus,
  MemoryType,
  Metadata,
  MetadataFilter,
  MetricKind,
  NextStepOptions,
  NodeEdgesResult,
  Procedure,
  ProcedureConstraint,
  ProcedureEdge,
  ProcedureNode,
  ProcedureStep,
  RerankRequest,
  RerankResult,
  ScrollOptions,
  ScrollResult,
  SearchHit,
  SearchQuery,
  SearchResult,
  SqlExecResult,
  SqlQueryResult,
  StateEntry,
  StatePutOptions,
  UpsertFactOptions,
  VectorItem,
  VectorUpdateRequest,
} from "./types.js";
