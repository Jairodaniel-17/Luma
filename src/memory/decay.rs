use crate::memory::service::MemoryService;

impl MemoryService {
    /// Apply exponential decay to all active semantic facts in all namespaces.
    ///
    /// `decay_score(t) = exp(-ln(2) / half_life_days * elapsed_days)`
    /// Facts whose score drops below `archive_threshold` are archived automatically.
    pub async fn run_decay_pass(&self) -> anyhow::Result<usize> {
        let Some(sqlite) = &self.sqlite else {
            return Ok(0);
        };

        let half_life_days = self.config.memory_decay_half_life_days;
        let threshold = self.config.memory_decay_archive_threshold;
        let now_ms = crate::memory::ingest::now_ms();

        // λ = ln(2) / half_life_days, converted to per-ms
        let lambda_per_ms = std::f64::consts::LN_2 / (half_life_days * 86_400_000.0);

        // Fetch all active/draft semantic facts with their creation timestamp
        let rows = sqlite
            .query(
                "SELECT id, namespace, created_at_ms, decay_score
                 FROM memory_records
                 WHERE kind = 'semantic' AND status IN ('active', 'draft')"
                    .to_string(),
                vec![],
            )
            .await?;

        let mut archived = 0usize;

        for row in rows {
            let id = row
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let namespace = row
                .get("namespace")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let created_at_ms = row
                .get("created_at_ms")
                .and_then(|v| v.as_u64())
                .unwrap_or(now_ms);

            let elapsed_ms = now_ms.saturating_sub(created_at_ms) as f64;
            let new_score = (-lambda_per_ms * elapsed_ms).exp() as f32;

            if new_score < threshold {
                // Archive the fact
                let _ = sqlite
                    .execute(
                        "UPDATE memory_records SET status = 'archived', decay_score = ?
                         WHERE namespace = ? AND id = ?"
                            .to_string(),
                        vec![
                            serde_json::json!(new_score),
                            serde_json::Value::String(namespace),
                            serde_json::Value::String(id),
                        ],
                    )
                    .await;
                archived += 1;
            } else {
                // Update decay_score only
                let _ = sqlite
                    .execute(
                        "UPDATE memory_records SET decay_score = ? WHERE namespace = ? AND id = ?"
                            .to_string(),
                        vec![
                            serde_json::json!(new_score),
                            serde_json::Value::String(namespace),
                            serde_json::Value::String(id),
                        ],
                    )
                    .await;
            }
        }

        if archived > 0 {
            tracing::info!(archived, "memory decay: archived facts below threshold");
        }
        Ok(archived)
    }

    /// Spawn a background task that runs `run_decay_pass` periodically.
    /// Returns immediately; the task runs until the process exits.
    pub fn spawn_decay_task(self: std::sync::Arc<Self>) {
        if !self.config.memory_decay_enabled {
            return;
        }
        let interval_secs = self.config.memory_decay_interval_secs;
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(std::time::Duration::from_secs(interval_secs));
            ticker.tick().await; // skip first immediate tick
            loop {
                ticker.tick().await;
                match self.run_decay_pass().await {
                    Ok(archived) => {
                        tracing::debug!(archived, "memory decay pass complete");
                    }
                    Err(e) => {
                        tracing::warn!("memory decay pass failed: {}", e);
                    }
                }
            }
        });
    }
}
