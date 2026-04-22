use crate::sqlite::SqliteService;

pub async fn ensure_memory_schema(sqlite: &SqliteService) -> anyhow::Result<()> {
    let statements = [
        "CREATE TABLE IF NOT EXISTS memory_records (
            id TEXT PRIMARY KEY,
            namespace TEXT NOT NULL,
            entity_id TEXT,
            kind TEXT NOT NULL,
            status TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata TEXT NOT NULL,
            confidence REAL NOT NULL,
            source TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL,
            expires_at_ms INTEGER,
            embedding_ref TEXT
        )",
        "CREATE INDEX IF NOT EXISTS idx_memory_records_namespace_kind_entity
            ON memory_records(namespace, kind, entity_id)",
        "CREATE INDEX IF NOT EXISTS idx_memory_records_namespace_created
            ON memory_records(namespace, created_at_ms DESC)",
        "CREATE TABLE IF NOT EXISTS procedures (
            procedure_id TEXT NOT NULL,
            namespace TEXT NOT NULL,
            name TEXT NOT NULL,
            version INTEGER NOT NULL,
            status TEXT NOT NULL,
            description TEXT,
            confidence REAL NOT NULL,
            source TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL,
            PRIMARY KEY (procedure_id, namespace, version)
        )",
        "CREATE INDEX IF NOT EXISTS idx_procedures_namespace_status
            ON procedures(namespace, status)",
        "CREATE TABLE IF NOT EXISTS procedure_nodes (
            procedure_id TEXT NOT NULL,
            namespace TEXT NOT NULL,
            version INTEGER NOT NULL,
            node_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            label TEXT NOT NULL,
            payload TEXT NOT NULL,
            PRIMARY KEY (procedure_id, namespace, version, node_id)
        )",
        "CREATE TABLE IF NOT EXISTS procedure_edges (
            procedure_id TEXT NOT NULL,
            namespace TEXT NOT NULL,
            version INTEGER NOT NULL,
            from_node_id TEXT NOT NULL,
            to_node_id TEXT NOT NULL,
            priority INTEGER NOT NULL,
            condition_json TEXT
        )",
        "CREATE INDEX IF NOT EXISTS idx_procedure_edges_lookup
            ON procedure_edges(procedure_id, namespace, version, from_node_id, priority DESC)",
        "CREATE TABLE IF NOT EXISTS procedure_constraints (
            constraint_id TEXT NOT NULL,
            procedure_id TEXT NOT NULL,
            namespace TEXT NOT NULL,
            version INTEGER NOT NULL,
            target_node_id TEXT,
            condition_json TEXT NOT NULL,
            message TEXT,
            PRIMARY KEY (constraint_id, procedure_id, namespace, version)
        )",
        "CREATE TABLE IF NOT EXISTS memory_versions (
            id TEXT PRIMARY KEY,
            memory_id TEXT NOT NULL,
            namespace TEXT NOT NULL,
            version INTEGER NOT NULL,
            snapshot_json TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL
        )",
        "CREATE TABLE IF NOT EXISTS memory_edges (
            id TEXT PRIMARY KEY,
            namespace TEXT NOT NULL,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            edge_type TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 1.0,
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at_ms INTEGER NOT NULL
        )",
        "CREATE INDEX IF NOT EXISTS idx_memory_edges_src
            ON memory_edges(namespace, source_id)",
        "CREATE INDEX IF NOT EXISTS idx_memory_edges_tgt
            ON memory_edges(namespace, target_id)",
        "CREATE INDEX IF NOT EXISTS idx_memory_edges_type
            ON memory_edges(namespace, edge_type)",
        "CREATE TABLE IF NOT EXISTS memory_history (
            id TEXT PRIMARY KEY,
            fact_key TEXT NOT NULL,
            namespace TEXT NOT NULL,
            entity_id TEXT,
            content TEXT NOT NULL,
            confidence REAL NOT NULL,
            status TEXT NOT NULL,
            superseded_by TEXT,
            valid_from INTEGER NOT NULL,
            valid_until INTEGER,
            created_at_ms INTEGER NOT NULL
        )",
        "CREATE INDEX IF NOT EXISTS idx_memory_history_key
            ON memory_history(namespace, fact_key)",
        "CREATE INDEX IF NOT EXISTS idx_memory_history_entity
            ON memory_history(namespace, entity_id)",
        "ALTER TABLE memory_records ADD COLUMN centrality_score REAL NOT NULL DEFAULT 0.0",
        "ALTER TABLE memory_records ADD COLUMN decay_score REAL NOT NULL DEFAULT 1.0",
    ];

    for statement in statements {
        let result = sqlite.execute(statement.to_string(), vec![]).await;
        if let Err(e) = result {
            let msg = e.to_string();
            if msg.contains("duplicate column") || msg.contains("already exists") {
                continue;
            }
            return Err(e);
        }
    }
    Ok(())
}
