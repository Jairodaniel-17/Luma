use crate::engine::{Engine, EngineError};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocRecord {
    pub id: String,
    pub doc: serde_json::Value,
    pub revision: u64,
}

pub fn put_doc(
    engine: &Engine,
    collection: &str,
    id: &str,
    doc: serde_json::Value,
) -> Result<DocRecord, EngineError> {
    let key = doc_key(collection, id);
    let previous = engine.get_state(&key);
    let stored = engine.put_state(key, doc.clone(), None, None)?;
    if let Some(prev) = previous {
        update_indexes_remove(engine, collection, id, &prev.value)?;
    }
    update_indexes_add(engine, collection, id, &doc)?;
    Ok(DocRecord {
        id: id.to_string(),
        doc,
        revision: stored.revision,
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
        doc: item.value,
        revision: item.revision,
    }))
}

pub fn delete_doc(engine: &Engine, collection: &str, id: &str) -> Result<bool, EngineError> {
    let key = doc_key(collection, id);
    let existing = engine.get_state(&key);
    let deleted = engine.delete_state(&key)?;
    if deleted {
        if let Some(prev) = existing {
            update_indexes_remove(engine, collection, id, &prev.value)?;
        }
    }
    Ok(deleted)
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
                    if doc_matches(&item.value, filter) {
                        docs.push(DocRecord {
                            id,
                            doc: item.value,
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
        if !doc_matches(&item.value, filter) {
            continue;
        }
        docs.push(DocRecord {
            id,
            doc: item.value,
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

fn update_indexes_add(
    engine: &Engine,
    collection: &str,
    id: &str,
    doc: &serde_json::Value,
) -> Result<(), EngineError> {
    for (field, value) in string_fields(doc) {
        modify_index(engine, &index_key(collection, &field, &value), |ids| {
            if !ids.iter().any(|existing| existing == id) {
                ids.push(id.to_string());
            }
        })?;
    }
    Ok(())
}

fn update_indexes_remove(
    engine: &Engine,
    collection: &str,
    id: &str,
    doc: &serde_json::Value,
) -> Result<(), EngineError> {
    for (field, value) in string_fields(doc) {
        modify_index(engine, &index_key(collection, &field, &value), |ids| {
            ids.retain(|existing| existing != id);
        })?;
    }
    Ok(())
}

fn modify_index<F>(engine: &Engine, key: &str, mut edit: F) -> Result<(), EngineError>
where
    F: FnMut(&mut Vec<String>),
{
    let state = engine.get_state(key);
    let mut ids = state
        .as_ref()
        .map(|item| parse_ids(&item.value))
        .unwrap_or_default();
    edit(&mut ids);
    ids.sort();
    ids.dedup();
    if ids.is_empty() {
        if state.is_some() {
            let _ = engine.delete_state(key)?;
        }
        return Ok(());
    }
    let value = serde_json::json!({ "ids": ids });
    let _ = engine.put_state(
        key.to_string(),
        value,
        None,
        state.as_ref().map(|item| item.revision),
    )?;
    Ok(())
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

fn parse_ids(value: &serde_json::Value) -> Vec<String> {
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
