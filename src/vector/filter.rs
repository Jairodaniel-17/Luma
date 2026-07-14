use crate::vector::VectorError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────

/// A composable metadata filter for vector search.
///
/// Accepted JSON forms:
/// ```json
/// // Leaf condition
/// { "field": "age", "op": "gte", "value": 18 }
///
/// // Values array (for 'in' / 'not_in')
/// { "field": "tag", "op": "in", "values": ["a", "b"] }
///
/// // Logical AND
/// { "and": [ {...}, {...} ] }
///
/// // Logical OR
/// { "or": [ {...}, {...} ] }
///
/// // Logical NOT
/// { "not": { "field": "deleted", "op": "eq", "value": true } }
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MetadataFilter {
    Condition(FilterCondition),
    And { and: Vec<MetadataFilter> },
    Or { or: Vec<MetadataFilter> },
    Not { not: Box<MetadataFilter> },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FilterCondition {
    pub field: String,
    pub op: FilterOp,
    /// Single value — used by Eq, Neq, Gt, Gte, Lt, Lte, Contains, StartsWith.
    #[serde(default)]
    pub value: Value,
    /// Array of values — used by In and NotIn. If empty, `value` is used as the
    /// array instead (accept both `"value": [...]` and `"values": [...]`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub values: Vec<Value>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FilterOp {
    /// == (works for strings, numbers, booleans, null)
    Eq,
    /// !=
    Neq,
    /// > (numeric or lexicographic for strings)
    Gt,
    /// >=
    Gte,
    /// <
    Lt,
    /// <=
    Lte,
    /// scalar field value is one of the provided values
    In,
    /// scalar field value is none of the provided values
    NotIn,
    /// array field contains at least one of the provided `values`
    /// (e.g. `tax_system: ["suitetax","legacy"]` matches `any_of: ["suitetax"]`)
    AnyOf,
    /// string field contains the substring
    Contains,
    /// string field starts with the given prefix
    StartsWith,
    /// field exists and is not null
    Exists,
}

// ─────────────────────────────────────────────────────────────────
// Evaluation
// ─────────────────────────────────────────────────────────────────

/// Evaluate `filter` against a metadata JSON object.
/// Returns `true` if the object matches, `false` otherwise.
pub fn evaluate_filter(meta: &Value, filter: &MetadataFilter) -> bool {
    match filter {
        MetadataFilter::Condition(c) => evaluate_condition(meta, c),
        MetadataFilter::And { and } => and.iter().all(|f| evaluate_filter(meta, f)),
        MetadataFilter::Or { or } => or.iter().any(|f| evaluate_filter(meta, f)),
        MetadataFilter::Not { not } => !evaluate_filter(meta, not),
    }
}

fn get_field<'a>(meta: &'a Value, field: &str) -> Option<&'a Value> {
    let mut cur = meta;
    for part in field.split('.') {
        cur = cur.get(part)?;
    }
    Some(cur)
}

fn compare_json(a: &Value, b: &Value) -> Option<Ordering> {
    match (a, b) {
        (Value::Number(an), Value::Number(bn)) => an.as_f64()?.partial_cmp(&bn.as_f64()?),
        (Value::String(as_), Value::String(bs)) => Some(as_.cmp(bs)),
        _ => None,
    }
}

fn in_values(actual: &Value, values: &[Value], fallback: &Value) -> bool {
    if values.is_empty() {
        if let Some(arr) = fallback.as_array() {
            return arr.iter().any(|v| v == actual);
        }
        return false;
    }
    values.iter().any(|v| v == actual)
}

/// Returns the effective query value list for `AnyOf`: prefers `values`, falls back to
/// treating `value` as an array or wrapping it in a single-element slice.
fn effective_query_values(cond: &FilterCondition) -> Vec<&Value> {
    if !cond.values.is_empty() {
        return cond.values.iter().collect();
    }
    if let Some(arr) = cond.value.as_array() {
        return arr.iter().collect();
    }
    if !cond.value.is_null() {
        return vec![&cond.value];
    }
    vec![]
}

fn evaluate_condition(meta: &Value, cond: &FilterCondition) -> bool {
    match cond.op {
        FilterOp::Exists => get_field(meta, &cond.field).is_some_and(|v| !v.is_null()),
        FilterOp::Eq => get_field(meta, &cond.field).is_some_and(|v| v == &cond.value),
        FilterOp::Neq => get_field(meta, &cond.field) != Some(&cond.value),
        FilterOp::Gt => get_field(meta, &cond.field)
            .and_then(|v| compare_json(v, &cond.value))
            .is_some_and(|o| o == Ordering::Greater),
        FilterOp::Gte => get_field(meta, &cond.field)
            .and_then(|v| compare_json(v, &cond.value))
            .is_some_and(|o| o != Ordering::Less),
        FilterOp::Lt => get_field(meta, &cond.field)
            .and_then(|v| compare_json(v, &cond.value))
            .is_some_and(|o| o == Ordering::Less),
        FilterOp::Lte => get_field(meta, &cond.field)
            .and_then(|v| compare_json(v, &cond.value))
            .is_some_and(|o| o != Ordering::Greater),
        FilterOp::In => {
            get_field(meta, &cond.field).is_some_and(|v| in_values(v, &cond.values, &cond.value))
        }
        FilterOp::NotIn => {
            get_field(meta, &cond.field).is_none_or(|v| !in_values(v, &cond.values, &cond.value))
        }
        FilterOp::AnyOf => {
            // Field must be a JSON array; true if any element matches any query value.
            let query_vals = effective_query_values(cond);
            get_field(meta, &cond.field)
                .and_then(|v| v.as_array())
                .is_some_and(|arr| arr.iter().any(|elem| query_vals.contains(&elem)))
        }
        FilterOp::Contains => get_field(meta, &cond.field)
            .and_then(|v| v.as_str())
            .zip(cond.value.as_str())
            .is_some_and(|(hay, needle)| hay.contains(needle)),
        FilterOp::StartsWith => get_field(meta, &cond.field)
            .and_then(|v| v.as_str())
            .zip(cond.value.as_str())
            .is_some_and(|(s, prefix)| s.starts_with(prefix)),
    }
}

// ─────────────────────────────────────────────────────────────────
// Keyword index fast path
// ─────────────────────────────────────────────────────────────────

/// Try to resolve a candidate ID set from the keyword index.
///
/// - Returns `Some(set)` when at least one `Eq(field, string)` condition in the
///   filter can be answered by the index. The set is an upper bound; callers must
///   still run `evaluate_filter` to confirm exact matches.
/// - Returns `None` when no conditions are index-resolvable (full scan needed).
///
/// Rules:
/// - `And`: intersects candidates from all index-resolvable children. Non-
///   indexable children still appear in the post-filter step.
/// - `Or`: unions candidates only when ALL children are index-resolvable.
///   If any child is not resolvable the result could be incomplete, so `None`
///   is returned and the full set is evaluated.
/// - `Not`, `Gt`, `Lt`, `Contains`, …: not resolvable via index.
pub fn index_candidates(
    filter: &MetadataFilter,
    index: &HashMap<String, HashMap<String, HashSet<Arc<str>>>>,
) -> Option<HashSet<String>> {
    match filter {
        MetadataFilter::Condition(c) if c.op == FilterOp::Eq => {
            let s = c.value.as_str()?;
            let ids = index.get(&c.field)?.get(s)?;
            Some(ids.iter().map(|a| a.to_string()).collect())
        }
        MetadataFilter::Condition(c) if c.op == FilterOp::AnyOf => {
            // Union of all indexed sets for each query value (strings only).
            let by_field = index.get(&c.field)?;
            let query_vals = effective_query_values(c);
            if query_vals.is_empty() {
                return Some(HashSet::new());
            }
            let mut union: HashSet<String> = HashSet::new();
            for qv in &query_vals {
                if let Some(s) = qv.as_str() {
                    if let Some(ids) = by_field.get(s) {
                        union.extend(ids.iter().map(|a| a.to_string()));
                    }
                    // If a query string has no entries, it contributes nothing to the union.
                }
            }
            Some(union)
        }
        MetadataFilter::And { and } => {
            let mut result: Option<HashSet<String>> = None;
            for sub in and {
                if let Some(candidates) = index_candidates(sub, index) {
                    result = Some(match result {
                        None => candidates,
                        Some(acc) => acc.intersection(&candidates).cloned().collect(),
                    });
                }
                // Non-resolvable children are handled by evaluate_filter; no action here.
            }
            result
        }
        MetadataFilter::Or { or } => {
            let mut union: HashSet<String> = HashSet::new();
            for sub in or {
                // Any unresolvable child → can't safely pre-filter (? returns None).
                union.extend(index_candidates(sub, index)?);
            }
            Some(union)
        }
        _ => None,
    }
}

// ─────────────────────────────────────────────────────────────────
// Backward-compatibility conversion
// ─────────────────────────────────────────────────────────────────

/// Convert the legacy flat-object filter `{"field": "value", ...}` (AND of exact
/// equality) to a typed `MetadataFilter`. Returns `None` for empty objects.
pub fn from_legacy(filters: &Value) -> Option<MetadataFilter> {
    let obj = filters.as_object()?;
    if obj.is_empty() {
        return None;
    }
    let mut conditions: Vec<MetadataFilter> = obj
        .iter()
        .map(|(k, v)| {
            MetadataFilter::Condition(FilterCondition {
                field: k.clone(),
                op: FilterOp::Eq,
                value: v.clone(),
                values: Vec::new(),
            })
        })
        .collect();
    if conditions.len() == 1 {
        conditions.pop()
    } else {
        Some(MetadataFilter::And { and: conditions })
    }
}

// ─────────────────────────────────────────────────────────────────
// SQL translation (for hub / SQLite pre-filtering)
// ─────────────────────────────────────────────────────────────────

/// Translate a `MetadataFilter` into a parameterized SQL WHERE clause fragment
/// suitable for `WHERE <sql>` on a table with a `metadata JSON` column.
///
/// Returns `(sql_fragment, params)`. Always use `json_extract(metadata, '$.field')`
/// so the query can benefit from SQLite JSON function indexes.
///
/// Field names are validated (see [`validate_field`]) because they are
/// interpolated directly into the SQL JSON path string and therefore cannot be
/// bound as parameters. A field containing anything outside the identifier-safe
/// character set is rejected with [`VectorError::InvalidFilterField`].
pub fn to_sql_where(filter: &MetadataFilter) -> Result<(String, Vec<Value>), VectorError> {
    let mut params = Vec::new();
    let sql = filter_to_sql(filter, &mut params)?;
    Ok((sql, params))
}

/// Validate a metadata field name that will be interpolated into a SQL JSON
/// path. Only identifier-safe characters are allowed: ASCII letters, digits,
/// underscore, dot (for nested paths) and hyphen. This rejects quotes,
/// parentheses, whitespace, semicolons and every other character an attacker
/// could use to break out of the `'$.<field>'` string literal.
fn validate_field(field: &str) -> Result<(), VectorError> {
    if field.is_empty()
        || !field
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-'))
    {
        return Err(VectorError::InvalidFilterField);
    }
    Ok(())
}

fn filter_to_sql(filter: &MetadataFilter, params: &mut Vec<Value>) -> Result<String, VectorError> {
    match filter {
        MetadataFilter::Condition(c) => condition_to_sql(c, params),
        MetadataFilter::And { and } => {
            if and.is_empty() {
                return Ok("1".to_string());
            }
            let parts = and
                .iter()
                .map(|f| filter_to_sql(f, params).map(|s| format!("({s})")))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(parts.join(" AND "))
        }
        MetadataFilter::Or { or } => {
            if or.is_empty() {
                return Ok("0".to_string());
            }
            let parts = or
                .iter()
                .map(|f| filter_to_sql(f, params).map(|s| format!("({s})")))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(parts.join(" OR "))
        }
        MetadataFilter::Not { not } => Ok(format!("NOT ({})", filter_to_sql(not, params)?)),
    }
}

fn condition_to_sql(
    cond: &FilterCondition,
    params: &mut Vec<Value>,
) -> Result<String, VectorError> {
    validate_field(&cond.field)?;
    let field_expr = format!("json_extract(metadata, '$.{}')", cond.field);
    let sql = match cond.op {
        FilterOp::Exists => format!("{field_expr} IS NOT NULL"),
        FilterOp::Eq => {
            params.push(cond.value.clone());
            format!("{field_expr} = ?")
        }
        FilterOp::Neq => {
            params.push(cond.value.clone());
            format!("{field_expr} != ?")
        }
        FilterOp::Gt => {
            params.push(cond.value.clone());
            format!("{field_expr} > ?")
        }
        FilterOp::Gte => {
            params.push(cond.value.clone());
            format!("{field_expr} >= ?")
        }
        FilterOp::Lt => {
            params.push(cond.value.clone());
            format!("{field_expr} < ?")
        }
        FilterOp::Lte => {
            params.push(cond.value.clone());
            format!("{field_expr} <= ?")
        }
        FilterOp::In => {
            let values = effective_values(cond);
            if values.is_empty() {
                return Ok("0".to_string());
            }
            let placeholders = std::iter::repeat_n("?", values.len())
                .collect::<Vec<_>>()
                .join(", ");
            params.extend(values.iter().cloned());
            format!("{field_expr} IN ({placeholders})")
        }
        FilterOp::NotIn => {
            let values = effective_values(cond);
            if values.is_empty() {
                return Ok("1".to_string());
            }
            let placeholders = std::iter::repeat_n("?", values.len())
                .collect::<Vec<_>>()
                .join(", ");
            params.extend(values.iter().cloned());
            format!("{field_expr} NOT IN ({placeholders})")
        }
        FilterOp::AnyOf => {
            // metadata column holds a JSON array; use json_each to check membership.
            // Generates: EXISTS (SELECT 1 FROM json_each(metadata, '$.field') WHERE value IN (?,...))
            let values = effective_values(cond);
            if values.is_empty() {
                return Ok("0".to_string());
            }
            let placeholders = std::iter::repeat_n("?", values.len())
                .collect::<Vec<_>>()
                .join(", ");
            params.extend(values.iter().cloned());
            // cond.field validated above.
            format!(
                "EXISTS (SELECT 1 FROM json_each(metadata, '$.{}') WHERE value IN ({placeholders}))",
                cond.field
            )
        }
        FilterOp::Contains => {
            params.push(cond.value.clone());
            format!("INSTR({field_expr}, ?) > 0")
        }
        FilterOp::StartsWith => {
            params.push(cond.value.clone());
            format!("INSTR({field_expr}, ?) = 1")
        }
    };
    Ok(sql)
}

fn effective_values(cond: &FilterCondition) -> &[Value] {
    if !cond.values.is_empty() {
        &cond.values
    } else if let Some(arr) = cond.value.as_array() {
        arr.as_slice()
    } else {
        &[]
    }
}

// ─────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn meta(v: serde_json::Value) -> serde_json::Value {
        v
    }

    fn cond(field: &str, op: FilterOp, value: serde_json::Value) -> MetadataFilter {
        MetadataFilter::Condition(FilterCondition {
            field: field.to_string(),
            op,
            value,
            values: Vec::new(),
        })
    }

    fn in_cond(field: &str, values: Vec<serde_json::Value>) -> MetadataFilter {
        MetadataFilter::Condition(FilterCondition {
            field: field.to_string(),
            op: FilterOp::In,
            value: Value::Null,
            values,
        })
    }

    // ── Leaf operators ──────────────────────────────────────────

    #[test]
    fn eq_matches_string() {
        let f = cond("status", FilterOp::Eq, json!("active"));
        assert!(evaluate_filter(&meta(json!({"status": "active"})), &f));
        assert!(!evaluate_filter(&meta(json!({"status": "inactive"})), &f));
    }

    #[test]
    fn eq_matches_number() {
        let f = cond("score", FilterOp::Eq, json!(42));
        assert!(evaluate_filter(&meta(json!({"score": 42})), &f));
        assert!(!evaluate_filter(&meta(json!({"score": 43})), &f));
    }

    #[test]
    fn neq_passes_when_field_missing() {
        let f = cond("x", FilterOp::Neq, json!("y"));
        assert!(evaluate_filter(&meta(json!({})), &f));
    }

    #[test]
    fn gt_gte_lt_lte_numbers() {
        let m = meta(json!({"n": 10}));
        assert!(evaluate_filter(&m, &cond("n", FilterOp::Gt, json!(9))));
        assert!(evaluate_filter(&m, &cond("n", FilterOp::Gte, json!(10))));
        assert!(evaluate_filter(&m, &cond("n", FilterOp::Lt, json!(11))));
        assert!(evaluate_filter(&m, &cond("n", FilterOp::Lte, json!(10))));
        assert!(!evaluate_filter(&m, &cond("n", FilterOp::Gt, json!(10))));
        assert!(!evaluate_filter(&m, &cond("n", FilterOp::Lt, json!(10))));
    }

    #[test]
    fn in_matches_values_array() {
        let f = in_cond("tier", vec![json!("gold"), json!("platinum")]);
        assert!(evaluate_filter(&meta(json!({"tier": "gold"})), &f));
        assert!(evaluate_filter(&meta(json!({"tier": "platinum"})), &f));
        assert!(!evaluate_filter(&meta(json!({"tier": "bronze"})), &f));
    }

    #[test]
    fn not_in_excludes_values() {
        let f = MetadataFilter::Condition(FilterCondition {
            field: "tier".to_string(),
            op: FilterOp::NotIn,
            value: Value::Null,
            values: vec![json!("banned"), json!("suspended")],
        });
        assert!(evaluate_filter(&meta(json!({"tier": "active"})), &f));
        assert!(!evaluate_filter(&meta(json!({"tier": "banned"})), &f));
    }

    #[test]
    fn contains_substring() {
        let f = cond("msg", FilterOp::Contains, json!("error"));
        assert!(evaluate_filter(
            &meta(json!({"msg": "fatal error occurred"})),
            &f
        ));
        assert!(!evaluate_filter(&meta(json!({"msg": "all good"})), &f));
    }

    #[test]
    fn starts_with_prefix() {
        let f = cond("path", FilterOp::StartsWith, json!("/api/"));
        assert!(evaluate_filter(&meta(json!({"path": "/api/users"})), &f));
        assert!(!evaluate_filter(&meta(json!({"path": "/web/home"})), &f));
    }

    #[test]
    fn exists_checks_non_null_presence() {
        let f = cond("field", FilterOp::Exists, json!(null));
        assert!(evaluate_filter(&meta(json!({"field": "value"})), &f));
        assert!(!evaluate_filter(&meta(json!({"field": null})), &f));
        assert!(!evaluate_filter(&meta(json!({})), &f));
    }

    #[test]
    fn dot_notation_nested_field() {
        let f = cond("a.b.c", FilterOp::Eq, json!("deep"));
        assert!(evaluate_filter(
            &meta(json!({"a": {"b": {"c": "deep"}}})),
            &f
        ));
        assert!(!evaluate_filter(
            &meta(json!({"a": {"b": {"c": "shallow"}}})),
            &f
        ));
    }

    // ── Logical operators ───────────────────────────────────────

    #[test]
    fn and_all_must_match() {
        let f = MetadataFilter::And {
            and: vec![
                cond("a", FilterOp::Eq, json!("x")),
                cond("b", FilterOp::Eq, json!("y")),
            ],
        };
        assert!(evaluate_filter(&meta(json!({"a": "x", "b": "y"})), &f));
        assert!(!evaluate_filter(&meta(json!({"a": "x", "b": "z"})), &f));
    }

    #[test]
    fn or_any_must_match() {
        let f = MetadataFilter::Or {
            or: vec![
                cond("status", FilterOp::Eq, json!("active")),
                cond("status", FilterOp::Eq, json!("pending")),
            ],
        };
        assert!(evaluate_filter(&meta(json!({"status": "active"})), &f));
        assert!(evaluate_filter(&meta(json!({"status": "pending"})), &f));
        assert!(!evaluate_filter(&meta(json!({"status": "inactive"})), &f));
    }

    #[test]
    fn not_negates() {
        let f = MetadataFilter::Not {
            not: Box::new(cond("deleted", FilterOp::Eq, json!(true))),
        };
        assert!(evaluate_filter(&meta(json!({"deleted": false})), &f));
        assert!(!evaluate_filter(&meta(json!({"deleted": true})), &f));
    }

    #[test]
    fn nested_and_or() {
        // (status == active) AND (score > 5 OR tier == gold)
        let f = MetadataFilter::And {
            and: vec![
                cond("status", FilterOp::Eq, json!("active")),
                MetadataFilter::Or {
                    or: vec![
                        cond("score", FilterOp::Gt, json!(5)),
                        cond("tier", FilterOp::Eq, json!("gold")),
                    ],
                },
            ],
        };
        assert!(evaluate_filter(
            &meta(json!({"status": "active", "score": 10})),
            &f
        ));
        assert!(evaluate_filter(
            &meta(json!({"status": "active", "tier": "gold"})),
            &f
        ));
        assert!(!evaluate_filter(
            &meta(json!({"status": "inactive", "score": 10})),
            &f
        ));
        assert!(!evaluate_filter(
            &meta(json!({"status": "active", "score": 3})),
            &f
        ));
    }

    // ── from_legacy ─────────────────────────────────────────────

    #[test]
    fn from_legacy_single_field() {
        let f = from_legacy(&json!({"status": "active"})).unwrap();
        assert!(evaluate_filter(&meta(json!({"status": "active"})), &f));
        assert!(!evaluate_filter(&meta(json!({"status": "inactive"})), &f));
    }

    #[test]
    fn from_legacy_multi_field_is_and() {
        let f = from_legacy(&json!({"a": "x", "b": "y"})).unwrap();
        assert!(evaluate_filter(&meta(json!({"a": "x", "b": "y"})), &f));
        assert!(!evaluate_filter(&meta(json!({"a": "x", "b": "z"})), &f));
    }

    #[test]
    fn from_legacy_empty_is_none() {
        assert!(from_legacy(&json!({})).is_none());
    }

    // ── index_candidates ────────────────────────────────────────

    fn build_index(
        entries: &[(&str, &str, &str)],
    ) -> HashMap<String, HashMap<String, HashSet<Arc<str>>>> {
        let mut idx: HashMap<String, HashMap<String, HashSet<Arc<str>>>> = HashMap::new();
        for (field, value, id) in entries {
            idx.entry(field.to_string())
                .or_default()
                .entry(value.to_string())
                .or_default()
                .insert(Arc::from(*id));
        }
        idx
    }

    #[test]
    fn index_candidates_eq_string() {
        let idx = build_index(&[
            ("status", "active", "id1"),
            ("status", "active", "id2"),
            ("status", "inactive", "id3"),
        ]);
        let f = cond("status", FilterOp::Eq, json!("active"));
        let candidates = index_candidates(&f, &idx).unwrap();
        assert!(candidates.contains("id1"));
        assert!(candidates.contains("id2"));
        assert!(!candidates.contains("id3"));
    }

    #[test]
    fn index_candidates_eq_non_string_returns_none() {
        let idx = build_index(&[]);
        let f = cond("score", FilterOp::Eq, json!(42));
        assert!(index_candidates(&f, &idx).is_none());
    }

    #[test]
    fn index_candidates_and_intersects() {
        let idx = build_index(&[
            ("status", "active", "id1"),
            ("status", "active", "id2"),
            ("tier", "gold", "id2"),
            ("tier", "gold", "id3"),
        ]);
        let f = MetadataFilter::And {
            and: vec![
                cond("status", FilterOp::Eq, json!("active")),
                cond("tier", FilterOp::Eq, json!("gold")),
            ],
        };
        let candidates = index_candidates(&f, &idx).unwrap();
        assert_eq!(candidates, HashSet::from(["id2".to_string()]));
    }

    #[test]
    fn index_candidates_or_unions_when_all_resolvable() {
        let idx = build_index(&[("status", "active", "id1"), ("status", "pending", "id2")]);
        let f = MetadataFilter::Or {
            or: vec![
                cond("status", FilterOp::Eq, json!("active")),
                cond("status", FilterOp::Eq, json!("pending")),
            ],
        };
        let candidates = index_candidates(&f, &idx).unwrap();
        assert_eq!(
            candidates,
            HashSet::from(["id1".to_string(), "id2".to_string()])
        );
    }

    #[test]
    fn index_candidates_or_with_non_indexable_returns_none() {
        let idx = build_index(&[("status", "active", "id1")]);
        let f = MetadataFilter::Or {
            or: vec![
                cond("status", FilterOp::Eq, json!("active")),
                cond("score", FilterOp::Gt, json!(5)), // not indexable
            ],
        };
        assert!(index_candidates(&f, &idx).is_none());
    }

    #[test]
    fn index_candidates_not_returns_none() {
        let idx = build_index(&[("status", "active", "id1")]);
        let f = MetadataFilter::Not {
            not: Box::new(cond("status", FilterOp::Eq, json!("active"))),
        };
        assert!(index_candidates(&f, &idx).is_none());
    }

    // ── SQL translation ─────────────────────────────────────────

    #[test]
    fn sql_eq_condition() {
        let f = cond("status", FilterOp::Eq, json!("active"));
        let (sql, params) = to_sql_where(&f).unwrap();
        assert!(sql.contains("json_extract(metadata, '$.status') = ?"));
        assert_eq!(params, vec![json!("active")]);
    }

    #[test]
    fn sql_in_condition() {
        let f = in_cond("tier", vec![json!("gold"), json!("platinum")]);
        let (sql, params) = to_sql_where(&f).unwrap();
        assert!(sql.contains("IN (?, ?)"));
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn sql_and_joins_with_and() {
        let f = MetadataFilter::And {
            and: vec![
                cond("a", FilterOp::Eq, json!("x")),
                cond("b", FilterOp::Gt, json!(5)),
            ],
        };
        let (sql, _) = to_sql_where(&f).unwrap();
        assert!(sql.contains(" AND "));
    }

    #[test]
    fn sql_or_joins_with_or() {
        let f = MetadataFilter::Or {
            or: vec![
                cond("a", FilterOp::Eq, json!("x")),
                cond("a", FilterOp::Eq, json!("y")),
            ],
        };
        let (sql, _) = to_sql_where(&f).unwrap();
        assert!(sql.contains(" OR "));
    }

    #[test]
    fn sql_not_wraps_with_not() {
        let f = MetadataFilter::Not {
            not: Box::new(cond("deleted", FilterOp::Eq, json!(true))),
        };
        let (sql, _) = to_sql_where(&f).unwrap();
        assert!(sql.starts_with("NOT ("));
    }

    #[test]
    fn sql_contains_uses_instr() {
        let f = cond("msg", FilterOp::Contains, json!("error"));
        let (sql, _) = to_sql_where(&f).unwrap();
        assert!(sql.contains("INSTR(") && sql.contains("> 0"));
    }

    // ── AnyOf (array membership) ────────────────────────────────

    fn any_of_cond(field: &str, values: Vec<serde_json::Value>) -> MetadataFilter {
        MetadataFilter::Condition(FilterCondition {
            field: field.to_string(),
            op: FilterOp::AnyOf,
            value: Value::Null,
            values,
        })
    }

    #[test]
    fn any_of_matches_single_value_in_array() {
        // Exact use case: tax_system: ["suitetax", "legacy"], query "suitetax"
        let f = any_of_cond("tax_system", vec![json!("suitetax")]);
        assert!(evaluate_filter(
            &meta(json!({"tax_system": ["suitetax", "legacy"]})),
            &f
        ));
        assert!(evaluate_filter(
            &meta(json!({"tax_system": ["suitetax"]})),
            &f
        ));
        assert!(!evaluate_filter(
            &meta(json!({"tax_system": ["legacy"]})),
            &f
        ));
        assert!(!evaluate_filter(&meta(json!({"tax_system": []})), &f));
    }

    #[test]
    fn any_of_multi_query_values() {
        // Match docs that have "v3" OR "suitetax" in their tax_system array
        let f = any_of_cond("tax_system", vec![json!("suitetax"), json!("v3")]);
        assert!(!evaluate_filter(
            &meta(json!({"tax_system": ["legacy"]})),
            &f
        ));
        assert!(evaluate_filter(&meta(json!({"tax_system": ["v3"]})), &f));
        assert!(evaluate_filter(
            &meta(json!({"tax_system": ["suitetax", "legacy"]})),
            &f
        ));
    }

    #[test]
    fn any_of_does_not_match_scalar_field() {
        // AnyOf requires the field to be an array; scalar string should not match
        let f = any_of_cond("tax_system", vec![json!("suitetax")]);
        assert!(!evaluate_filter(
            &meta(json!({"tax_system": "suitetax"})),
            &f
        ));
    }

    #[test]
    fn any_of_missing_field_returns_false() {
        let f = any_of_cond("tax_system", vec![json!("suitetax")]);
        assert!(!evaluate_filter(&meta(json!({})), &f));
    }

    #[test]
    fn any_of_index_candidates_fast_path() {
        // Verify the keyword index fast path resolves AnyOf correctly
        let mut index: HashMap<String, HashMap<String, HashSet<Arc<str>>>> = HashMap::new();
        // Simulate indexing two docs:
        //   doc1: tax_system: ["suitetax", "legacy"]
        //   doc2: tax_system: ["legacy"]
        index
            .entry("tax_system".to_string())
            .or_default()
            .entry("suitetax".to_string())
            .or_default()
            .insert(Arc::from("doc1"));
        index
            .entry("tax_system".to_string())
            .or_default()
            .entry("legacy".to_string())
            .or_default()
            .insert(Arc::from("doc1"));
        index
            .entry("tax_system".to_string())
            .or_default()
            .entry("legacy".to_string())
            .or_default()
            .insert(Arc::from("doc2"));

        let f = any_of_cond("tax_system", vec![json!("suitetax")]);
        let candidates = index_candidates(&f, &index).unwrap();
        assert!(candidates.contains("doc1"));
        assert!(!candidates.contains("doc2"));
    }

    #[test]
    fn any_of_and_eq_combines_correctly() {
        // AND: country == "pe" AND tax_system AnyOf ["suitetax"]
        let f = MetadataFilter::And {
            and: vec![
                cond("country", FilterOp::Eq, json!("pe")),
                any_of_cond("tax_system", vec![json!("suitetax")]),
            ],
        };
        assert!(evaluate_filter(
            &meta(json!({"country": "pe", "tax_system": ["suitetax", "legacy"]})),
            &f
        ));
        assert!(!evaluate_filter(
            &meta(json!({"country": "pe", "tax_system": ["legacy"]})),
            &f
        ));
        assert!(!evaluate_filter(
            &meta(json!({"country": "br", "tax_system": ["suitetax"]})),
            &f
        ));
    }

    #[test]
    fn sql_any_of_uses_json_each() {
        let f = any_of_cond("tax_system", vec![json!("suitetax"), json!("legacy")]);
        let (sql, params) = to_sql_where(&f).unwrap();
        assert!(sql.contains("json_each"), "should use json_each: {sql}");
        assert!(
            sql.contains("IN (?, ?)"),
            "should have 2 placeholders: {sql}"
        );
        assert_eq!(params.len(), 2);
    }

    // ── Field-name injection ────────────────────────────────────

    #[test]
    fn sql_rejects_injection_field() {
        // A field name attempting to break out of the '$.<field>' JSON path
        // and inject SQL must be rejected, not interpolated.
        let malicious = "x') UNION SELECT password FROM users --";
        let f = cond(malicious, FilterOp::Eq, json!("y"));
        assert!(matches!(
            to_sql_where(&f),
            Err(VectorError::InvalidFilterField)
        ));

        // Rejected inside AnyOf (the other interpolation site) too.
        let f = any_of_cond("a'b", vec![json!("z")]);
        assert!(matches!(
            to_sql_where(&f),
            Err(VectorError::InvalidFilterField)
        ));

        // And when nested inside a logical combinator.
        let f = MetadataFilter::And {
            and: vec![
                cond("ok_field", FilterOp::Eq, json!("v")),
                cond("bad field", FilterOp::Eq, json!("v")),
            ],
        };
        assert!(matches!(
            to_sql_where(&f),
            Err(VectorError::InvalidFilterField)
        ));
    }

    #[test]
    fn sql_accepts_identifier_safe_fields() {
        // Dotted nested paths, digits, underscores and hyphens are allowed.
        for field in ["status", "a.b.c", "field_1", "kebab-case", "n0.n1_x"] {
            let f = cond(field, FilterOp::Eq, json!("v"));
            assert!(
                to_sql_where(&f).is_ok(),
                "field should be accepted: {field}"
            );
        }
    }
}
