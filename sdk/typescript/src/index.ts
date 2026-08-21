// ─── Main client ──────────────────────────────────────────────────────────────
export { LumaClient } from "./client.js";

// ─── Sub-clients (tree-shakeable named exports) ───────────────────────────────
export { AccountsClient } from "./accounts.js";
export { AdminClient } from "./admin.js";
export { BlobClient } from "./blob.js";
export { ConfigClient } from "./config.js";
export { AuthClient } from "./auth.js";
export { DiskAnnClient } from "./diskann.js";
export { DocClient } from "./doc.js";
export { EventsClient } from "./events.js";
export { HubClient } from "./hub.js";
export { MemoryClient } from "./memory.js";
export { QueueClient } from "./queue.js";
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
  AccessPolicy,
  AggregateRequest,
  AggregateResult,
  ApiKeyInfo,
  AuditEntry,
  AuditEventEntry,
  AuditEventList,
  AuditQueryOptions,
  BackupResult,
  BatchPutOperation,
  BatchSearchResult,
  BeliefVersion,
  BlobListResult,
  BlobPutResult,
  CentralityResult,
  CollectionDetail,
  CollectionInfo,
  CreateEdgeRequest,
  CreateKeyResult,
  DiskAnnBuildOptions,
  DiskAnnStatus,
  DocFindOptions,
  DocFindResult,
  DomainOrgList,
  DomainOrgMapping,
  EdgeType,
  EmbeddingProbeRequest,
  EmbeddingProbeResult,
  EventStreamOptions,
  FilterOp,
  HubIngestOptions,
  HubSearchHit,
  HubSearchOptions,
  HubSearchResult,
  ImageTransformOptions,
  IngestEventOptions,
  InviteResult,
  JsonValue,
  KeyRole,
  LumaClientOptions,
  LumaEvent,
  Member,
  MemberList,
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
  Organization,
  OrgList,
  Permission,
  PermissionCheck,
  PermissionList,
  Procedure,
  ProcedureConstraint,
  ProcedureEdge,
  ProcedureNode,
  ProcedureStep,
  QueueAck,
  QueueEnqueueResult,
  QueueMessage,
  QueueReceiveResult,
  QueueStats,
  RefreshResult,
  ReindexProgress,
  ReindexStart,
  RerankRequest,
  RerankResult,
  Role,
  RoleList,
  ScrollOptions,
  ScrollResult,
  SearchHit,
  SearchQuery,
  SearchResult,
  SessionInfo,
  SessionList,
  SessionResult,
  StateEntry,
  StatePutOptions,
  SwitchOrgResult,
  UpsertFactOptions,
  User,
  UserList,
  VectorItem,
  VectorUpdateRequest,
} from "./types.js";
