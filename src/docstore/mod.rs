use crate::engine::{Engine, EngineError, StateError, StateItem, StateOp};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocRecord {
    pub id: String,
    pub doc: serde_json::Value,
    pub revision: u64,
}

/// How many times a document write starts over because the document changed
/// between the unlocked read that picks the index keys and the locked write.
/// Every retry means someone else committed, so the store as a whole always
/// makes progress; the bound only turns a pathological storm into an error.
const MAX_ATTEMPTS: usize = 1000;

/// Write a document and its index entries as **one** atomic, durable unit.
///
/// The index keys depend on the document's current fields, so they are picked
/// from an unlocked read; then the document and all of them are locked and
/// read again (`Engine::update_states`). If the document is no longer what was
/// read, its fields may point at other index keys, and the write starts over.
/// Under the locks nothing else can touch these keys, so the index update is
/// exact. It used to be separate writes, with index entries read-modify-written
/// unguarded, so concurrent writers to one field value lost each other's ids
/// and a crash could leave a document and its index disagreeing.
pub fn put_doc(
    engine: &Engine,
    collection: &str,
    id: &str,
    doc: serde_json::Value,
) -> Result<DocRecord, EngineError> {
    write_doc(engine, collection, id, Some(doc.clone())).map(|revision| DocRecord {
        id: id.to_string(),
        doc,
        revision,
    })
}

pub fn get_doc(
    engine: &Engine,
    collection: &str,
    id: &str,
) -> Result<Option<DocRecord>, EngineError> {
    let key = doc_key(collection, id);
    Ok(engine.get_state(&key).map(|item| DocRecord {
        id: id.to_string(),
        // The doc store is JSON-only: a raw byte value under a doc: key has
        // no document representation, so it reads as null rather than being
        // silently reinterpreted.
        doc: item.value.into_json().unwrap_or(serde_json::Value::Null),
        revision: item.revision,
    }))
}

/// Delete a document and drop it from its index entries, atomically, like
/// `put_doc`.
pub fn delete_doc(engine: &Engine, collection: &str, id: &str) -> Result<bool, EngineError> {
    write_doc(engine, collection, id, None).map(|revision| revision > 0)
}

/// Put (`Some`) or delete (`None`) a document with its index entries. Returns
/// the document's new revision, or for a delete 1 if it existed and 0 if not.
fn write_doc(
    engine: &Engine,
    collection: &str,
    id: &str,
    doc: Option<serde_json::Value>,
) -> Result<u64, EngineError> {
    let key = doc_key(collection, id);
    let new_fields = doc.as_ref().map(string_fields).unwrap_or_default();
    for _ in 0..MAX_ATTEMPTS {
        let seen = engine.get_state(&key);
        let old_fields = fields_of(seen.as_ref());
        // Index key -> whether the id belongs in it after this write.
        let mut wanted: BTreeMap<String, bool> = BTreeMap::new();
        for (field, value) in &old_fields {
            wanted.insert(index_key(collection, field, value), false);
        }
        for (field, value) in &new_fields {
            wanted.insert(index_key(collection, field, value), true);
        }
        let mut keys = vec![key.clone()];
        keys.extend(wanted.keys().cloned());

        let done = engine.update_states(&keys, |items| {
            let current = items[0].as_ref();
            // Same fields as the unlocked read, so `keys` covers every index
            // entry this write touches. Otherwise start over.
            if fields_of(current) != old_fields {
                return Ok::<_, EngineError>((Vec::new(), None));
            }
            let mut ops = Vec::with_capacity(keys.len());
            let revision = match (&doc, current) {
                (Some(doc), _) => {
                    ops.push(StateOp::Put {
                        key: key.clone(),
                        value: doc.clone().into(),
                        ttl_ms: None,
                        if_revision: None,
                    });
                    current.map_or(1, |item| item.revision.saturating_add(1))
                }
                (None, Some(_)) => {
                    ops.push(StateOp::Delete {
                        key: key.clone(),
                        if_revision: None,
                    });
                    1
                }
                (None, None) => return Ok((Vec::new(), Some(0))),
            };
            for (index, item) in keys[1..].iter().zip(&items[1..]) {
                let mut ids = item
                    .as_ref()
                    .map(|i| parse_ids(&i.value))
                    .unwrap_or_default();
                ids.retain(|existing| existing != id);
                if wanted[index] {
                    ids.push(id.to_string());
                }
                ids.sort();
                ids.dedup();
                if ids.is_empty() {
                    if item.is_some() {
                        ops.push(StateOp::Delete {
                            key: index.clone(),
                            if_revision: None,
                        });
                    }
                } else {
                    ops.push(StateOp::Put {
                        key: index.clone(),
                        value: serde_json::json!({ "ids": ids }).into(),
                        ttl_ms: None,
                        if_revision: None,
                    });
                }
            }
            Ok((ops, Some(revision)))
        })?;
        if let Some(revision) = done {
            return Ok(revision);
        }
        std::thread::yield_now();
    }
    Err(StateError::RevisionMismatch.into())
}

/// The index format the stored entries are known to match. Version 2: before
/// it, concurrent writers lost each other's ids, so entries written by an
/// older Luma can be missing documents and must be rebuilt.
const INDEX_VERSION: u64 = 2;
const INDEX_VERSION_KEY: &str = "docmeta:index_version";

/// Rebuild every document index from the documents themselves, once per
/// `INDEX_VERSION`. Run at startup, before anything is served, so nothing
/// writes concurrently. Returns how many documents were reindexed.
///
/// This is the migration for data written by a Luma with the lost-id bug: the
/// documents are intact, only their index entries may not be.
pub fn migrate_indexes(engine: &Engine) -> Result<usize, EngineError> {
    let current = engine
        .get_state(INDEX_VERSION_KEY)
        .and_then(|item| item.value.as_json().and_then(|v| v.as_u64()));
    if current >= Some(INDEX_VERSION) {
        return Ok(0);
    }

    let stale: Vec<String> = scan(engine, "docidx:").map(|item| item.key).collect();
    for chunk in stale.chunks(500) {
        let ops = chunk
            .iter()
            .map(|key| StateOp::Delete {
                key: key.clone(),
                if_revision: None,
            })
            .collect();
        engine.put_state_batch(ops)?;
    }

    let mut wanted: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut docs = 0;
    for item in scan(engine, "doc:") {
        // Collections cannot contain ':', so the first one after the prefix
        // ends the collection; the id may contain more.
        let Some((collection, id)) = item.key["doc:".len()..].split_once(':') else {
            continue;
        };
        docs += 1;
        for (field, value) in fields_of(Some(&item)) {
            wanted
                .entry(index_key(collection, &field, &value))
                .or_default()
                .push(id.to_string());
        }
    }
    let entries: Vec<(String, Vec<String>)> = wanted.into_iter().collect();
    for chunk in entries.chunks(500) {
        let ops = chunk
            .iter()
            .map(|(key, ids)| {
                let mut ids = ids.clone();
                ids.sort();
                ids.dedup();
                StateOp::Put {
                    key: key.clone(),
                    value: serde_json::json!({ "ids": ids }).into(),
                    ttl_ms: None,
                    if_revision: None,
                }
            })
            .collect();
        engine.put_state_batch(ops)?;
    }
    engine.put_state(
        INDEX_VERSION_KEY.to_string(),
        serde_json::json!(INDEX_VERSION),
        None,
        None,
    )?;
    Ok(docs)
}

/// Every live item whose key starts with `prefix`, page by page.
fn scan<'a>(engine: &'a Engine, prefix: &'a str) -> impl Iterator<Item = StateItem> + 'a {
    // The first key past every key that starts with `prefix`.
    let mut end = prefix.to_string();
    let last = end.pop().map(|c| char::from_u32(c as u32 + 1).unwrap_or(c));
    end.extend(last);
    let mut start = Some(prefix.to_string());
    let mut page = Vec::new().into_iter();
    std::iter::from_fn(move || loop {
        if let Some(item) = page.next() {
            return Some(item);
        }
        let from = start.take()?;
        let items = engine.list_state_range(Some(&from), Some(&end), 1000);
        if let Some(last) = items.last() {
            start = Some(format!("{}\0", last.key));
        }
        page = items.into_iter();
        if page.len() == 0 {
            return None;
        }
    })
}

fn fields_of(item: Option<&StateItem>) -> Vec<(String, String)> {
    item.and_then(|i| i.value.as_json())
        .map(string_fields)
        .unwrap_or_default()
}

pub fn find_docs(
    engine: &Engine,
    collection: &str,
    filter: Option<&serde_json::Value>,
    limit: usize,
) -> Result<Vec<DocRecord>, EngineError> {
    let limit = limit.max(1);
    if let Some(f) = filter {
        if let Some(ids) = indexed_candidates(engine, collection, f)? {
            if ids.is_empty() {
                return Ok(Vec::new());
            }
            let mut docs = Vec::new();
            for id in ids {
                if docs.len() >= limit {
                    break;
                }
                if let Some(item) = engine.get_state(&doc_key(collection, &id)) {
                    if item
                        .value
                        .as_json()
                        .is_some_and(|json| doc_matches(json, filter))
                    {
                        docs.push(DocRecord {
                            id,
                            doc: item.value.into_json().unwrap_or(serde_json::Value::Null),
                            revision: item.revision,
                        });
                    }
                }
            }
            return Ok(docs);
        }
    }

    let prefix = format!("doc:{collection}:");
    let mut docs = Vec::new();
    for item in engine.list_state(Some(&prefix), limit * 4) {
        if !item.key.starts_with(&prefix) {
            continue;
        }
        let id = item.key[prefix.len()..].to_string();
        let Some(json) = item.value.as_json() else {
            continue;
        };
        if !doc_matches(json, filter) {
            continue;
        }
        docs.push(DocRecord {
            id,
            doc: item.value.into_json().unwrap_or(serde_json::Value::Null),
            revision: item.revision,
        });
        if docs.len() >= limit {
            break;
        }
    }
    Ok(docs)
}

fn doc_key(collection: &str, id: &str) -> String {
    format!("doc:{collection}:{id}")
}

fn index_key(collection: &str, field: &str, value: &str) -> String {
    format!("docidx:{collection}:{field}:{value}")
}

fn string_fields(doc: &serde_json::Value) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let Some(obj) = doc.as_object() else {
        return out;
    };
    for (k, v) in obj {
        if let Some(value) = v.as_str() {
            out.push((k.clone(), value.to_string()));
        }
    }
    out
}

/// Read the id list out of an index entry.
///
/// Takes the stored value rather than JSON so a raw byte value under an index
/// key yields an empty list instead of forcing every caller to unwrap.
fn parse_ids(value: &crate::engine::stored::StoredVal) -> Vec<String> {
    value
        .get("ids")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default()
}

fn indexed_candidates(
    engine: &Engine,
    collection: &str,
    filter: &serde_json::Value,
) -> Result<Option<Vec<String>>, EngineError> {
    let obj = match filter.as_object() {
        Some(o) => o,
        None => return Ok(None),
    };
    let mut result: Option<HashSet<String>> = None;
    for (field, value) in obj {
        let Some(value_str) = value.as_str() else {
            return Ok(None);
        };
        let key = index_key(collection, field, value_str);
        let Some(item) = engine.get_state(&key) else {
            return Ok(Some(Vec::new()));
        };
        let ids = parse_ids(&item.value);
        let set: HashSet<String> = ids.into_iter().collect();
        result = Some(match result {
            None => set,
            Some(current) => current.into_iter().filter(|id| set.contains(id)).collect(),
        });
    }
    Ok(result.map(|set| {
        let mut ids: Vec<String> = set.into_iter().collect();
        ids.sort();
        ids
    }))
}

fn doc_matches(doc: &serde_json::Value, filter: Option<&serde_json::Value>) -> bool {
    let Some(filter) = filter else { return true };
    let serde_json::Value::Object(filter_obj) = filter else {
        return false;
    };
    let Some(doc_obj) = doc.as_object() else {
        return false;
    };
    for (field, expected) in filter_obj {
        match doc_obj.get(field) {
            Some(actual) if actual == expected => {}
            _ => return false,
        }
    }
    true
}
