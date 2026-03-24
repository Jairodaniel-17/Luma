use crate::memory::service::MemoryService;
use crate::memory::types::MemoryQueryMode;

impl MemoryService {
    pub(crate) fn resolve_query_mode(
        &self,
        query: &str,
        requested: Option<MemoryQueryMode>,
    ) -> MemoryQueryMode {
        if let Some(mode) = requested {
            if mode != MemoryQueryMode::Auto {
                return mode;
            }
        }

        let query = query.to_ascii_lowercase();
        if query.contains("next step")
            || query.contains("siguiente paso")
            || query.contains("qué sigue")
            || query.contains("que sigue")
        {
            return MemoryQueryMode::NextStep;
        }
        if query.contains("timeline")
            || query.contains("historial")
            || query.contains("qué pasó")
            || query.contains("que paso")
            || query.contains("what happened")
        {
            return MemoryQueryMode::Timeline;
        }
        MemoryQueryMode::Recall
    }
}
