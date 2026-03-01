use crate::search::types::DocumentMetadata;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GroupKey {
    String(String),
    U32(u32),
    Unique(u64), // For items that don't belong to any group (e.g. missing field)
}

pub fn extract_key(meta: &DocumentMetadata, field: &str) -> Option<GroupKey> {
    match field {
        "group_id" => meta.group_id.map(GroupKey::U32),
        "document_id" => meta.document_id.clone().map(GroupKey::String),
        "filename" => meta.filename.clone().map(GroupKey::String),
        "category" => meta.category.clone().map(GroupKey::String),
        "language" => meta.language.clone().map(GroupKey::String),
        "status" => meta.status.clone().map(GroupKey::String),
        "version" => meta.version.clone().map(GroupKey::String),
        _ => None,
    }
}

// Internal struct to hold score and offset for grouping
#[derive(Debug, Clone, Copy)]
pub struct ScoredOffset {
    pub score: f32,
    pub offset: u64,
}

impl PartialEq for ScoredOffset {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}
impl Eq for ScoredOffset {}
impl PartialOrd for ScoredOffset {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ScoredOffset {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
    }
}

pub struct GroupedResults {
    // Map from GroupKey to a min-heap of best hits (Reverse(ScoredOffset))
    // We use min-heap to keep top-N.
    groups: HashMap<GroupKey, BinaryHeap<Reverse<ScoredOffset>>>,
    limit: usize,
}

impl GroupedResults {
    pub fn new(limit: usize) -> Self {
        Self {
            groups: HashMap::new(),
            limit,
        }
    }

    pub fn push(&mut self, key: GroupKey, score: f32, offset: u64) {
        let entry = self.groups.entry(key).or_default();

        entry.push(Reverse(ScoredOffset { score, offset }));
        if entry.len() > self.limit {
            entry.pop();
        }
    }

    // Convert to a flat list of best chunks, sorted by their score (ranked by group score)
    pub fn into_sorted_vec(self) -> Vec<(f32, u64)> {
        let mut group_leaders = Vec::new();

        for (_, heap) in self.groups {
            // heap contains top `limit` items.
            let items: Vec<_> = heap.into_iter().map(|Reverse(x)| x).collect();

            if items.is_empty() {
                continue;
            }

            // Get max score of the group (first item in sorted list)
            // Sort items desc
            let mut sorted_items = items;
            sorted_items.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

            let max_score = sorted_items[0].score;

            group_leaders.push((max_score, sorted_items));
        }

        // Sort groups by max_score desc
        group_leaders.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

        // Flatten
        let mut final_results = Vec::new();
        for (_, items) in group_leaders {
            for item in items {
                final_results.push((item.score, item.offset));
            }
        }

        final_results
    }
}
