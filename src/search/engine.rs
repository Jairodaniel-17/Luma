use crate::search::grouping::{extract_key, GroupKey, GroupedResults};
use crate::search::storage::AppendLog;
use crate::search::types::{
    Document, DocumentResponse, LanguageFilter, SearchRequest, SearchResponse, SearchResult,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use std::path::PathBuf;

pub struct SearchEngine {
    storage: AppendLog,
}

impl SearchEngine {
    pub fn new(data_dir: PathBuf) -> anyhow::Result<Self> {
        let path = data_dir.join("search").join("documents.log");
        Ok(Self {
            storage: AppendLog::new(path)?,
        })
    }

    pub fn ingest(&self, doc: Document) -> anyhow::Result<()> {
        self.storage.append(&doc)?;
        Ok(())
    }

    pub fn search(&self, req: SearchRequest) -> anyhow::Result<SearchResponse> {
        // 1. Embed
        // Assume 384 dimensions for now.
        let query_vector = self.embed(&req.query, 384);

        // 2. Filter & Version Resolution
        // Map group_id -> (offset, processed_at, grouping_key)
        let mut candidates = HashMap::new();
        let mut ungroupped_candidates = Vec::new(); // (offset, grouping_key)

        // Default to "all" if not specified.
        let version_policy = req
            .filters
            .as_ref()
            .and_then(|f| f.version_policy.as_deref())
            .unwrap_or("all");

        let is_latest = version_policy == "latest";
        let group_field = req.group_by.as_deref();

        let iter = self.storage.scan_metadata()?;
        for res in iter {
            let (offset, _id, meta) = res?;

            // Filters
            if let Some(filters) = &req.filters {
                if let Some(c) = &filters.category {
                    if meta.category.as_ref() != Some(c) {
                        continue;
                    }
                }
                if let Some(s) = &filters.status {
                    if meta.status.as_ref() != Some(s) {
                        continue;
                    }
                }
                if let Some(lang_filter) = &filters.language {
                    match lang_filter {
                        LanguageFilter::Single(l) => {
                            if meta.language.as_ref() != Some(l) {
                                continue;
                            }
                        }
                        LanguageFilter::Multiple(langs) => {
                            if let Some(l) = &meta.language {
                                if !langs.contains(l) {
                                    continue;
                                }
                            } else {
                                continue;
                            }
                        }
                    }
                }
            }

            let group_key = group_field.and_then(|f| extract_key(&meta, f));

            if is_latest {
                if let Some(gid) = meta.group_id {
                    let pat = meta.processed_at.unwrap_or(0);
                    candidates
                        .entry(gid)
                        .and_modify(|(off, old_pat, old_key)| {
                            if pat > *old_pat {
                                *off = offset;
                                *old_pat = pat;
                                *old_key = group_key.clone();
                            }
                        })
                        .or_insert((offset, pat, group_key));
                } else {
                    ungroupped_candidates.push((offset, group_key));
                }
            } else {
                ungroupped_candidates.push((offset, group_key));
            }
        }

        let mut final_candidates = ungroupped_candidates;
        if is_latest {
            for (_, (off, _, key)) in candidates {
                final_candidates.push((off, key));
            }
        }

        // 3. Score
        let mut results_vec: Vec<(f32, u64)> = Vec::new();

        if group_field.is_some() {
            let mut grouped = GroupedResults::new(req.group_limit);

            for (offset, key) in final_candidates {
                let vec = self.storage.read_vector(offset)?;
                if vec.len() != query_vector.len() {
                    continue;
                }
                let score = cosine_similarity(&query_vector, &vec);

                // If key is None (field missing), treat as unique group
                let k = key.unwrap_or(GroupKey::Unique(offset));
                grouped.push(k, score, offset);
            }
            results_vec = grouped.into_sorted_vec();
        } else {
            // Min-heap of top-k results (stores Reverse(ScoredDoc) so smallest score is popped)
            let mut heap = BinaryHeap::with_capacity(req.top_k + 1);

            for (offset, _) in final_candidates {
                let vec = self.storage.read_vector(offset)?;
                if vec.len() != query_vector.len() {
                    continue;
                }
                let score = cosine_similarity(&query_vector, &vec);

                heap.push(Reverse(ScoredDoc { score, offset }));
                if heap.len() > req.top_k {
                    heap.pop();
                }
            }
            let sorted_docs = heap.into_sorted_vec();
            for Reverse(doc) in sorted_docs {
                results_vec.push((doc.score, doc.offset));
            }
        }

        // 4. Retrieve
        // Truncate to top_k if needed (GroupedResults already sorted by group score, but we might have more than top_k items if group_limit > 1)
        // Wait, the requirement says "múltiples chunks del mismo documento no saturen el top-k, retornando un solo resultado por grupo".
        // This means we want `top_k` GROUPS.
        // But `GroupedResults::into_sorted_vec` returns a flat list.
        // Does it return `top_k` groups?
        // `GroupedResults` stores ALL groups that were pushed. It doesn't know about `top_k`.
        // So `results_vec` contains ALL groups (limited by `group_limit` per group).
        // So we need to take `top_k` from `results_vec`.
        // However, if `group_limit` > 1, taking `top_k` items might slice a group?
        // "group_limit: 1". "top_k: 10". Result: 10 groups, 1 item each.
        // "group_limit: 2". "top_k: 10". Result: 10 groups, 2 items each? Or 10 items total?
        // Usually top_k refers to the number of *results* returned.
        // The spec says:
        // "Test 2 ... 3 docs, 3 chunks each ... top_k=10 ... resultado: 3 hits"
        // This implies top_k is a limit on the response list size.
        // If I have 100 groups, and top_k=10, I return 10 groups.
        // If I have 5 groups, and top_k=10, I return 5 groups.
        // If group_limit=2, and 5 groups, I return 10 items.
        // So I should truncate `results_vec` to `req.top_k`?
        // Or should I truncate to `req.top_k` *groups*?
        // "diversidad del top-k aumenta".
        // "no saturen el top-k".
        // Usually top-k implies K items.
        // I will assume `top_k` is the number of results to return.

        let final_results_iter = results_vec.into_iter().take(req.top_k);

        let mut results = Vec::new();
        for (score, offset) in final_results_iter {
            let full_doc = self.storage.read_document(offset)?;
            results.push(SearchResult {
                score,
                document: DocumentResponse {
                    id: full_doc.id,
                    content: full_doc.content,
                    metadata: full_doc.metadata,
                },
            });
        }

        Ok(SearchResponse {
            query: req.query,
            top_k: req.top_k,
            results,
        })
    }

    fn embed(&self, text: &str, dim: usize) -> Vec<f32> {
        if let Some(stripped) = text.strip_prefix("TEST_VEC:") {
            let parts: Vec<&str> = stripped.split(',').collect();
            if let Ok(vec) = parts
                .iter()
                .map(|s| s.trim().parse::<f32>())
                .collect::<Result<Vec<_>, _>>()
            {
                if !vec.is_empty() {
                    return vec;
                }
            }
        }

        let hash = crc32fast::hash(text.as_bytes());
        let mut rng = StdRng::seed_from_u64(hash as u64);
        (0..dim).map(|_| rng.gen::<f32>()).collect()
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

struct ScoredDoc {
    score: f32,
    offset: u64,
}

impl PartialEq for ScoredDoc {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}
impl Eq for ScoredDoc {}
impl PartialOrd for ScoredDoc {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ScoredDoc {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
    }
}
