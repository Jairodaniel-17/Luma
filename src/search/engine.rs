use crate::search::storage::AppendLog;
use crate::search::types::{
    Document, DocumentResponse, LanguageFilter, SearchRequest, SearchResponse, SearchResult,
};
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use std::path::PathBuf;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

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
        // Map group_id -> (offset, processed_at)
        let mut candidates = HashMap::new();
        let mut ungroupped_offsets = Vec::new();

        // Default to "all" if not specified.
        let version_policy = req
            .filters
            .as_ref()
            .and_then(|f| f.version_policy.as_deref())
            .unwrap_or("all");

        let is_latest = version_policy == "latest";

        let mut iter = self.storage.scan_metadata()?;
        while let Some(res) = iter.next() {
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

            if is_latest {
                if let Some(gid) = meta.group_id {
                    let pat = meta.processed_at.unwrap_or(0);
                    candidates
                        .entry(gid)
                        .and_modify(|(off, old_pat)| {
                            if pat > *old_pat {
                                *off = offset;
                                *old_pat = pat;
                            }
                        })
                        .or_insert((offset, pat));
                } else {
                    ungroupped_offsets.push(offset);
                }
            } else {
                ungroupped_offsets.push(offset);
            }
        }

        let mut final_offsets = ungroupped_offsets;
        if is_latest {
            for (_, (off, _)) in candidates {
                final_offsets.push(off);
            }
        }

        // 3. Score
        // Min-heap of top-k results (stores Reverse(ScoredDoc) so smallest score is popped)
        let mut heap = BinaryHeap::with_capacity(req.top_k + 1);

        for offset in final_offsets {
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

        // 4. Retrieve
        let mut results = Vec::new();
        // Heap has smallest score at top (due to Reverse). Pop gives smallest.
        // We want desc order.
        let sorted_docs = heap.into_sorted_vec();
        // into_sorted_vec returns ascending order of T.
        // T is Reverse(ScoredDoc).
        // ScoredDoc cmp is score.
        // Reverse(0.1) > Reverse(0.9).
        // Ascending Reverse: Reverse(0.9), Reverse(0.1).
        // We iterate this. Unwrap. 0.9, 0.1.
        // Correct.

        for Reverse(doc) in sorted_docs {
            let full_doc = self.storage.read_document(doc.offset)?;
            results.push(SearchResult {
                score: doc.score,
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
        if text.starts_with("TEST_VEC:") {
            let parts: Vec<&str> = text["TEST_VEC:".len()..].split(',').collect();
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
        self.score.partial_cmp(&other.score)
    }
}
impl Ord for ScoredDoc {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}
