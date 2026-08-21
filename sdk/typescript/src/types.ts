// ─── Shared primitive types ───────────────────────────────────────────────────

export type MetricKind = "cosine" | "euclidean" | "dot";

export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

export type Metadata = Record<string, JsonValue>;

// ─── Vector types ─────────────────────────────────────────────────────────────

export interface CollectionInfo {
  name: string;
  dim: number;
  metric: MetricKind;
  count: number;
}

export interface CollectionDetail extends CollectionInfo {
  segments: number;
  index_kind: string;
}

export interface VectorItem {
  id: string;
  vector?: number[];
  meta?: Metadata;
}

export interface MetadataFilter {
  [field: string]: JsonValue | FilterOp;
}

export interface FilterOp {
  eq?: JsonValue;
  neq?: JsonValue;
  gt?: number;
  gte?: number;
  lt?: number;
  lte?: number;
  in?: JsonValue[];
  not_in?: JsonValue[];
  any_of?: JsonValue[];
  contains?: string;
  starts_with?: string;
  exists?: boolean;
  and?: MetadataFilter[];
  or?: MetadataFilter[];
  not?: MetadataFilter;
}

export interface SearchQuery {
  vector: number[];
  k: number;
  filters?: MetadataFilter;
  filter?: MetadataFilter;
  include_meta?: boolean;
}

export interface SearchHit {
  id: string;
  score: number;
  meta?: Metadata;
  vector?: number[];
}

export interface SearchResult {
  hits: SearchHit[];
}

export interface BatchSearchResult {
  results: SearchResult[];
}

export interface ScrollOptions {
  cursor?: string;
  limit?: number;
  include_vectors?: boolean;
}

export interface ScrollResult {
  items: VectorItem[];
  next_cursor?: string;
}

export interface RerankRequest {
  ids: string[];
  query_text?: string;
  query_vector?: number[];
}

export interface RerankResult {
  items: Array<{ id: string; score: number }>;
}

export interface AggregateRequest {
  group_by: string;
  filter?: MetadataFilter;
  limit?: number;
}

export interface AggregateResult {
  buckets: Array<{ value: JsonValue; count: number }>;
}

export interface VectorUpdateRequest {
  id: string;
  vector?: number[];
  meta?: Metadata;
}

// ─── State (KV) types ─────────────────────────────────────────────────────────

export interface StateEntry {
  key: string;
  value: JsonValue;
  revision: number;
  expires_at?: number;
}

export interface StatePutOptions {
  ttl_ms?: number;
  if_revision?: number;
}

export interface BatchPutOperation {
  key: string;
  value: JsonValue;
  ttl_ms?: number;
}

// ─── Document store types ─────────────────────────────────────────────────────

export interface DocFindOptions {
  filter?: Metadata;
  limit?: number;
}

export interface DocFindResult {
  documents: Array<{ id: string; doc: Metadata }>;
}

// ─── Events / SSE types ───────────────────────────────────────────────────────

export interface LumaEvent {
  offset: number;
  kind: string;
  payload: JsonValue;
  ts_ms: number;
}

export interface EventStreamOptions {
  since?: number;
  types?: string;
  key_prefix?: string;
  collection?: string;
}

// ─── Hub (LumaDatabase) types ─────────────────────────────────────────────────

export interface HubIngestOptions {
  id?: string;
  metadata?: Metadata;
}

export interface HubSearchOptions {
  sql_filter?: string;
  limit?: number;
}

export interface HubSearchHit {
  id: string;
  score: number;
  text?: string;
  metadata?: Metadata;
}

export interface HubSearchResult {
  hits: HubSearchHit[];
}

// ─── Memory / NS-Mem types ────────────────────────────────────────────────────

export type MemoryStatus = "active" | "draft" | "archived";
export type MemoryType = "episodic" | "semantic" | "procedural" | "working";
export type EdgeType =
  | "triggered_by"
  | "supports"
  | "contradicts"
  | "supersedes"
  | "related_to"
  | string;

export interface MemoryRecord {
  id: string;
  memory_type: MemoryType;
  content: string;
  metadata?: Metadata;
  entity_id?: string;
  source?: string;
  session_id?: string;
  fact_key?: string;
  confidence?: number;
  status?: MemoryStatus;
  centrality_score?: number;
  decay_score?: number;
  created_at_ms: number;
  updated_at_ms?: number;
}

export interface IngestEventOptions {
  id?: string;
  entity_id?: string;
  metadata?: Metadata;
  source?: string;
  session_id?: string;
  expires_at_ms?: number;
}

export interface UpsertFactOptions {
  id?: string;
  entity_id?: string;
  fact_key?: string;
  metadata?: Metadata;
  source?: string;
  confidence?: number;
  status?: MemoryStatus;
}

export interface MemoryQueryOptions {
  entity_id?: string;
  session_id?: string;
  procedure_id?: string;
  current_node_id?: string;
  context?: Metadata;
  mode?: string;
  limit?: number;
  include_evidence?: boolean;
  include_plan?: boolean;
  include_diagnostics?: boolean;
}

export interface MemoryQueryResult {
  records: MemoryRecord[];
  evidence?: MemoryRecord[];
  plan?: ProcedureStep[];
  diagnostics?: Record<string, JsonValue>;
}

export interface MemoryEdge {
  id: string;
  source_id: string;
  target_id: string;
  edge_type: EdgeType;
  weight?: number;
  metadata?: Metadata;
  created_at_ms: number;
}

export interface CreateEdgeRequest {
  source_id: string;
  target_id: string;
  edge_type: EdgeType;
  id?: string;
  weight?: number;
  metadata?: Metadata;
}

export interface BeliefVersion {
  version: number;
  content: string;
  metadata?: Metadata;
  confidence?: number;
  status?: MemoryStatus;
  recorded_at_ms: number;
}

export interface ProcedureNode {
  id: string;
  name: string;
  description?: string;
  metadata?: Metadata;
}

export interface ProcedureEdge {
  from_id: string;
  to_id: string;
  edge_type: string;
  condition?: string;
}

export interface ProcedureConstraint {
  node_id: string;
  expression: string;
}

export interface Procedure {
  procedure_id: string;
  name: string;
  nodes: ProcedureNode[];
  edges: ProcedureEdge[];
  version?: number;
  status?: string;
  description?: string;
  confidence?: number;
  source?: string;
  constraints?: ProcedureConstraint[];
}

export interface NextStepOptions {
  current_node_id?: string;
  context?: Metadata;
}

export interface ProcedureStep {
  node_id: string;
  name: string;
  description?: string;
  metadata?: Metadata;
}

export interface NodeEdgesResult {
  outgoing: MemoryEdge[];
  incoming: MemoryEdge[];
}

export interface CentralityResult {
  updated: number;
}

// ─── Auth types ───────────────────────────────────────────────────────────────

export type KeyRole = "admin" | "user";

export interface ApiKeyInfo {
  id: string;
  name: string;
  role: KeyRole;
  created_at_ms: number;
  revoked?: boolean;
}

export interface CreateKeyResult extends ApiKeyInfo {
  key: string;
}

// ─── Admin types ──────────────────────────────────────────────────────────────

export interface BackupResult {
  ok: boolean;
  offset: number;
}

export interface AuditEntry {
  ts: number;
  api_key_id: string;
  ip: string;
  method: string;
  path: string;
  status: number;
  latency_ms: number;
}

export interface AuditQueryOptions {
  from_ms?: number;
  to_ms?: number;
  key?: string;
  limit?: number;
}

// ─── DiskANN types ────────────────────────────────────────────────────────────

export interface DiskAnnBuildOptions {
  max_degree?: number;
  build_threads?: number;
  search_list_size?: number;
}

export interface DiskAnnStatus {
  built: boolean;
  node_count?: number;
  max_degree?: number;
  search_list_size?: number;
}

// ─── Accounts, organizations and members ──────────────────────────────────────

export interface SessionResult {
  /** Opaque session token, prefixed `lums_`. Only its SHA-256 hash is stored. */
  token: string;
  expires_at_ms: number;
  role: string;
  org_id?: string | null;
  user_id?: string;
}

export interface RefreshResult {
  token: string;
  expires_at_ms: number;
}

export interface SwitchOrgResult {
  token: string;
  expires_at_ms: number;
  org_id: string;
  role: string;
}

/** Session metadata. The token itself is never returned. */
export interface SessionInfo {
  created_at_ms?: number;
  expires_at_ms?: number;
  org_id?: string | null;
  current?: boolean;
}

export interface SessionList {
  sessions: SessionInfo[];
  count: number;
}

export interface Organization {
  id: string;
  name: string;
  /** The caller's role in this org, when listing memberships. */
  role?: string;
  created_at_ms?: number;
}

export interface OrgList {
  orgs: Organization[];
}

export interface Member {
  user_id: string;
  email: string;
  role: string;
}

export interface MemberList {
  members: Member[];
}

export interface InviteResult {
  user_id: string;
  email: string;
  org_id: string;
  role: string;
  /** True when the account did not exist and was created. */
  created: boolean;
  /** Present only when generated. Returned once, never retrievable again. */
  temp_password?: string | null;
}

export interface User {
  id: string;
  email: string;
  role: string;
  org_id?: string | null;
}

export interface UserList {
  users: User[];
}

/** Allow-list governing who may self-register. */
export interface AccessPolicy {
  domains?: string[];
  emails?: string[];
}

export interface DomainOrgMapping {
  domain: string;
  org_id: string;
  role: string;
}

export interface DomainOrgList {
  mappings: DomainOrgMapping[];
}

/** Business-level event from sys_audit_events. */
export interface AuditEventEntry {
  ts?: number;
  org_id?: string | null;
  user_id?: string | null;
  action?: string;
  target?: string | null;
  detail?: unknown;
}

export interface AuditEventList {
  events: AuditEventEntry[];
  count: number;
}

// ─── Roles and permissions ────────────────────────────────────────────────────

export interface Role {
  id: string;
  name: string;
  parent_role_id?: string | null;
  description?: string | null;
}

export interface RoleList {
  roles: Role[];
}

export interface Permission {
  resource: string;
  action: string;
}

export interface PermissionList {
  permissions: Permission[];
}

export interface PermissionCheck extends Permission {
  role: string;
  allowed: boolean;
}

// ─── Object storage and images ────────────────────────────────────────────────

export interface BlobPutResult {
  bucket: string;
  key: string;
  /** Stored object size in bytes. */
  size: number;
  etag: string;
}

export interface BlobListResult {
  keys: string[];
  count: number;
}

export interface ImageTransformOptions {
  w?: number;
  h?: number;
  format?: "png" | "jpeg";
  /** JPEG only; ignored for png. */
  quality?: number;
}

// ─── Queues ───────────────────────────────────────────────────────────────────

export interface QueueEnqueueResult {
  id: string;
}

export interface QueueMessage {
  id: string;
  body: unknown;
  /** Delivery attempts so far, including this one. Above 1 means a lease expired. */
  attempts: number;
}

export interface QueueReceiveResult {
  messages: QueueMessage[];
}

export interface QueueAck {
  ok: boolean;
}

export interface QueueStats {
  queue: string;
  /** Total messages held, leased ones included. */
  depth: number;
  /** Messages currently available to receive. */
  visible: number;
}

// ─── Embedding probe ──────────────────────────────────────────────────────────

export interface EmbeddingProbeRequest {
  provider: string;
  url?: string;
  api_key?: string;
  model?: string;
  azure_api_base?: string;
  azure_deployment?: string;
  azure_api_version?: string;
}

export interface EmbeddingProbeResult {
  ok: boolean;
  /** Dimension actually returned by the provider. Present when ok is true. */
  dim?: number;
  provider?: string;
  model?: string;
  /** The provider's own error message. Present when ok is false. */
  error?: string;
}

// ─── Client options ───────────────────────────────────────────────────────────

export interface LumaClientOptions {
  baseUrl: string;
  apiKey: string;
  timeout?: number;
}
