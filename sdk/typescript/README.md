# luma-vdb

Official TypeScript/JavaScript SDK for **Luma** — the convergent data engine that unifies vector search, key-value state, relational SQL, and pub/sub events in a single binary.

## Installation

```bash
npm install luma-vdb
# or
yarn add luma-vdb
# or
pnpm add luma-vdb
```

Requires Node.js 18+ (native `fetch` required). Also works in modern browsers and edge runtimes (Cloudflare Workers, Deno, Bun).

## Quick start

```typescript
import { LumaClient } from 'luma-vdb';

const client = new LumaClient({
  baseUrl: 'http://localhost:1234',
  apiKey: 'my-secret-key',
  timeout: 30_000, // optional, default 30s
});
```

## Vector operations

```typescript
// Create a collection
await client.vector.createCollection('docs', 384, 'cosine');

// Upsert a single vector
await client.vector.upsert('docs', 'item-1', [0.1, 0.2, /* ... */], {
  category: 'tech',
  source: 'web',
});

// Upsert many vectors at once
await client.vector.upsertBatch('docs', [
  { id: 'a', vector: [0.1, 0.2], meta: { tag: 'ai' } },
  { id: 'b', vector: [0.3, 0.4], meta: { tag: 'ml' } },
]);

// Nearest-neighbour search
const results = await client.vector.search('docs', {
  vector: [0.1, 0.2, /* ... */],
  k: 10,
  filter: { category: { eq: 'tech' } },
  include_meta: true,
});
console.log(results.hits); // [{ id, score, meta }]

// Batch search (up to 100 queries in parallel)
const batch = await client.vector.searchBatch('docs', [
  { vector: [0.1, 0.2], k: 5 },
  { vector: [0.3, 0.4], k: 5 },
]);
console.log(batch.results); // [{hits: [...]}, {hits: [...]}]

// Scroll through all vectors (cursor-based pagination)
let cursor: string | undefined;
do {
  const page = await client.vector.scroll('docs', { cursor, limit: 100 });
  for (const item of page.items) console.log(item.id);
  cursor = page.next_cursor;
} while (cursor);

// Re-rank a candidate list by cosine similarity
const reranked = await client.vector.rerank('docs', {
  ids: ['a', 'b', 'c'],
  query_text: 'machine learning',
});

// Aggregate by metadata field
const stats = await client.vector.aggregate('docs', {
  group_by: 'category',
  limit: 20,
});
console.log(stats.buckets); // [{ value: 'tech', count: 42 }, ...]
```

## Key-value state store

```typescript
// Set a key with optional TTL and optimistic locking
await client.state.put('user:123:prefs', { theme: 'dark' });
await client.state.put('session:abc', { uid: 1 }, { ttl_ms: 60_000 });

// Compare-and-swap: raises LumaConflictError if revision doesn't match
await client.state.put('counter', 10, { if_revision: 5 });

// Get, list, delete
const entry = await client.state.get('user:123:prefs');
const keys  = await client.state.list('user:123:', 50);
await client.state.delete('session:abc');

// Batch write
await client.state.batchPut([
  { key: 'a', value: 1 },
  { key: 'b', value: 2, ttl_ms: 5_000 },
]);

// Secondary index (in-memory, recreate after restart)
await client.state.createIndex('role');
const admins = await client.state.queryIndex('role', 'admin');
```

## Document store

```typescript
await client.doc.put('profiles', 'user-1', { name: 'Alice', plan: 'pro' });
const profile = await client.doc.get('profiles', 'user-1');
await client.doc.delete('profiles', 'user-1');

// Find by filter
const results = await client.doc.find('profiles', {
  filter: { plan: 'pro' },
  limit: 50,
});
```

## SQL (SQLite)

```typescript
// SELECT — returns rows + column names
const { rows } = await client.sql.query(
  'SELECT id, name FROM users WHERE active = ? LIMIT ?',
  [true, 10],
);

// DDL / DML — returns rows_affected
const { rows_affected } = await client.sql.exec(
  'UPDATE users SET last_seen = ? WHERE id = ?',
  [Date.now(), 'user-1'],
);
```

## LumaDatabase hybrid layer

The `db()` namespace handles auto-chunking, embedding generation, and hybrid SQL+vector search behind a simple two-method API.

```typescript
const db = client.db('my-app');

// Ingest a document — chunked, embedded, and indexed automatically
await db.ingest('The quick brown fox jumps over the lazy dog.', {
  id: 'doc-1',
  metadata: { source: 'web', author: 'Alice' },
});

// Hybrid search: SQL pre-filter + vector similarity
const hits = await db.search('fast animals', {
  sql_filter: "metadata->>'source' = 'web'",
  limit: 5,
});
console.log(hits.hits);
```

## Agent memory (NS-Mem)

NS-Mem provides four memory types for AI agents: episodic events, semantic facts, procedural DAGs, and ephemeral working memory.

```typescript
const memory = client.memory('agent-1');

// Episodic event — triggers LLM consolidation pipeline when enabled
await memory.ingestEvent('user asked about pricing', {
  entity_id: 'user-123',
  source: 'chat',
});

// Semantic fact
await memory.upsertFact('enterprise plan', {
  fact_key: 'user_plan',
  entity_id: 'user-123',
  confidence: 0.95,
});

// Query across all memory types
const recalled = await memory.query('pricing preferences', {
  entity_id: 'user-123',
  limit: 10,
  include_evidence: true,
});
console.log(recalled.records);

// Entity timeline
const history = await memory.timeline('user-123');

// Procedural DAG
await memory.upsertProcedure({
  procedure_id: 'onboarding',
  name: 'User Onboarding',
  nodes: [
    { id: 'start', name: 'Welcome' },
    { id: 'setup', name: 'Configure profile' },
    { id: 'done',  name: 'Complete' },
  ],
  edges: [
    { from_id: 'start', to_id: 'setup', edge_type: 'next' },
    { from_id: 'setup', to_id: 'done',  edge_type: 'next' },
  ],
});

const next = await memory.nextStep('onboarding', { current_node_id: 'start' });

// Graph edges
await memory.createEdge({
  source_id: 'fact-a',
  target_id: 'fact-b',
  edge_type: 'supports',
  weight: 0.9,
});
const edges = await memory.nodeEdges('fact-a');

// Belief versioning
const versions = await memory.beliefHistory('user_plan');

// PageRank centrality refresh
await memory.refreshCentrality();
```

## SSE event streaming

```typescript
// Streams all events from offset 0
for await (const event of client.events.stream()) {
  console.log(event.kind, event.payload);
}

// Stream with filters
for await (const event of client.events.stream({
  since: 1000,
  types: 'vector_upsert',
  collection: 'docs',
})) {
  console.log(event);
}
```

## Auth (admin only)

```typescript
// Requires an admin API key
const keys = await client.auth.listKeys();
const newKey = await client.auth.createKey('ci-pipeline', 'user');
console.log(newKey.key); // shown once
await client.auth.revokeKey(newKey.id);
```

## Admin

```typescript
// Trigger snapshot + WAL rotation (returns current offset)
const { offset } = await client.admin.backup();

// Query the audit log
const entries = await client.admin.audit({
  from_ms: Date.now() - 3_600_000,
  limit: 500,
});
```

## DiskANN index management

```typescript
const da = client.diskann('large-collection');

await da.build({ max_degree: 64, build_threads: 4 });
await da.tune({ search_list_size: 128 });
const status = await da.status();
console.log(status.built, status.node_count);
```

## Error handling

```typescript
import {
  LumaAuthError,
  LumaConflictError,
  LumaError,
  LumaForbiddenError,
  LumaNotFoundError,
} from 'luma-vdb';

try {
  await client.state.get('missing-key');
} catch (err) {
  if (err instanceof LumaNotFoundError) {
    // 404
  } else if (err instanceof LumaAuthError) {
    // 401 — bad or missing API key
  } else if (err instanceof LumaForbiddenError) {
    // 403 — insufficient role
  } else if (err instanceof LumaConflictError) {
    // 409 — CAS revision mismatch
  } else if (err instanceof LumaError) {
    console.error(err.status, err.body);
  }
}
```

## TypeScript types

All request and response shapes are fully typed. Import them from the package root:

```typescript
import type {
  CollectionInfo,
  LumaClientOptions,
  MemoryRecord,
  MetadataFilter,
  SearchHit,
  VectorItem,
} from 'luma-vdb';
```

## License

MIT
