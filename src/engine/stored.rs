//! The value type of the key-value store.
//!
//! F0.1 of `docs/PLAN-MAESTRO.md`. The KV store held `serde_json::Value`, which
//! cannot represent arbitrary bytes: Redis `SET key <binary>` and any
//! `application/octet-stream` payload had nowhere to go. [`StoredVal`] adds a
//! bytes variant beside the JSON one.
//!
//! ## Why the encoding looks like this
//!
//! Rule 1 of the data compatibility policy: every version reads what the
//! previous one wrote. Millions of existing records — in the JSON-lines WAL, in
//! redb, in snapshots — are bare JSON values, and none of them may be
//! reinterpreted.
//!
//! So `Json` serializes **transparently**: `StoredVal::Json(v)` produces exactly
//! `v`, byte for byte, and any value that is not a raw marker deserializes back
//! to `Json`. Old records keep their old meaning with no migration step, and a
//! downgrade still reads everything written before the first raw value appears.
//!
//! `Raw` serializes as a tagged object:
//!
//! ```json
//! { "__luma_raw": "<base64>", "content_type": "image/png" }
//! ```
//!
//! [`RAW_MARKER`] is therefore a reserved key. A caller storing JSON that
//! contains it at the top level would be handing us something indistinguishable
//! from a raw value, so [`StoredVal::from_json_checked`] rejects it rather than
//! letting the ambiguity through — a confusing error at write time beats a value
//! that changes type on the way back out.
//!
//! Bytes travel base64 because the WAL is JSON lines. That costs a third more
//! space on the wire and on disk; in memory the bytes are held decoded, so reads
//! do not pay for it repeatedly. A binary WAL record would remove the overhead
//! entirely and is the follow-up if it ever shows up in a profile.

use base64::Engine as _;
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Reserved top-level key that marks a raw byte payload.
pub const RAW_MARKER: &str = "__luma_raw";
const CONTENT_TYPE_KEY: &str = "content_type";

/// A value stored under a key.
#[derive(Clone, Debug, PartialEq)]
pub enum StoredVal {
    /// A JSON document. Serializes to itself, so existing records are untouched.
    Json(serde_json::Value),
    /// Opaque bytes with an optional media type.
    Raw {
        bytes: Vec<u8>,
        content_type: Option<String>,
    },
}

impl StoredVal {
    pub fn raw(bytes: Vec<u8>, content_type: Option<String>) -> Self {
        StoredVal::Raw {
            bytes,
            content_type,
        }
    }

    /// Wrap a JSON value, refusing one that would be ambiguous with a raw
    /// payload.
    ///
    /// Only a top-level object carrying [`RAW_MARKER`] is refused; the key is
    /// harmless anywhere deeper, because only the top level is inspected when
    /// decoding.
    pub fn from_json_checked(value: serde_json::Value) -> Result<Self, &'static str> {
        if let serde_json::Value::Object(map) = &value {
            if map.contains_key(RAW_MARKER) {
                return Err(
                    "`__luma_raw` is a reserved top-level key: a JSON object containing it \
                     cannot be told apart from a raw byte value",
                );
            }
        }
        Ok(StoredVal::Json(value))
    }

    /// The JSON document, when this is one. `None` for raw bytes — which is what
    /// makes a raw value simply unindexable by the secondary index rather than a
    /// special case there.
    pub fn as_json(&self) -> Option<&serde_json::Value> {
        match self {
            StoredVal::Json(value) => Some(value),
            StoredVal::Raw { .. } => None,
        }
    }

    /// Field lookup on the JSON document, mirroring `serde_json::Value::get`.
    ///
    /// `None` for raw bytes, which have no fields — so every caller that reads a
    /// field keeps working unchanged and simply sees nothing for a raw value.
    pub fn get(&self, key: &str) -> Option<&serde_json::Value> {
        self.as_json()?.get(key)
    }

    /// Take the JSON document, discarding a raw value.
    ///
    /// For boundaries that are JSON-only by contract — the doc store and the
    /// hub's metadata projections — where a raw value has no representation and
    /// being absent is the honest answer.
    pub fn into_json(self) -> Option<serde_json::Value> {
        match self {
            StoredVal::Json(value) => Some(value),
            StoredVal::Raw { .. } => None,
        }
    }

    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            StoredVal::Raw { bytes, .. } => Some(bytes),
            StoredVal::Json(_) => None,
        }
    }

    pub fn content_type(&self) -> Option<&str> {
        match self {
            StoredVal::Raw { content_type, .. } => content_type.as_deref(),
            StoredVal::Json(_) => None,
        }
    }

    pub fn is_raw(&self) -> bool {
        matches!(self, StoredVal::Raw { .. })
    }

    /// Size in bytes, for quota accounting. JSON is measured by its serialized
    /// length, which is what actually occupies the WAL.
    pub fn size_hint(&self) -> usize {
        match self {
            StoredVal::Raw { bytes, .. } => bytes.len(),
            StoredVal::Json(value) => serde_json::to_vec(value).map(|v| v.len()).unwrap_or(0),
        }
    }
}

impl Default for StoredVal {
    fn default() -> Self {
        StoredVal::Json(serde_json::Value::Null)
    }
}

/// Any JSON value becomes `Json` — the conversion every existing caller needs.
/// Use [`StoredVal::from_json_checked`] at trust boundaries where a caller could
/// supply the reserved marker.
impl From<serde_json::Value> for StoredVal {
    fn from(value: serde_json::Value) -> Self {
        StoredVal::Json(value)
    }
}

impl Serialize for StoredVal {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            // Transparent on purpose: this is what keeps every record written
            // before this type existed byte-identical.
            StoredVal::Json(value) => value.serialize(serializer),
            StoredVal::Raw {
                bytes,
                content_type,
            } => {
                let mut map = serde_json::Map::with_capacity(2);
                map.insert(
                    RAW_MARKER.to_string(),
                    serde_json::Value::String(
                        base64::engine::general_purpose::STANDARD.encode(bytes),
                    ),
                );
                if let Some(ct) = content_type {
                    map.insert(
                        CONTENT_TYPE_KEY.to_string(),
                        serde_json::Value::String(ct.clone()),
                    );
                }
                serde_json::Value::Object(map).serialize(serializer)
            }
        }
    }
}

impl<'de> Deserialize<'de> for StoredVal {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        if let serde_json::Value::Object(map) = &value {
            if let Some(encoded) = map.get(RAW_MARKER) {
                let encoded = encoded.as_str().ok_or_else(|| {
                    D::Error::custom(format!("`{RAW_MARKER}` must hold a base64 string"))
                })?;
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(encoded)
                    .map_err(|e| D::Error::custom(format!("`{RAW_MARKER}` is not base64: {e}")))?;
                let content_type = map
                    .get(CONTENT_TYPE_KEY)
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                return Ok(StoredVal::Raw {
                    bytes,
                    content_type,
                });
            }
        }
        // Everything else is a JSON document, which is what makes a legacy
        // record load with its original meaning.
        Ok(StoredVal::Json(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn roundtrip(value: &StoredVal) -> StoredVal {
        let text = serde_json::to_string(value).unwrap();
        serde_json::from_str(&text).unwrap()
    }

    #[test]
    fn json_serializes_transparently() {
        // The compatibility guarantee in one assertion: a JSON value stored
        // through StoredVal is byte-identical to the same value stored before
        // this type existed, so no WAL record needs rewriting.
        let value = json!({ "a": 1, "b": ["x", null, true] });
        let wrapped = StoredVal::Json(value.clone());
        assert_eq!(
            serde_json::to_string(&wrapped).unwrap(),
            serde_json::to_string(&value).unwrap()
        );
    }

    #[test]
    fn a_legacy_record_deserializes_as_json() {
        for text in [
            r#"{"a":1}"#,
            "null",
            "42",
            r#""a string""#,
            "[1,2,3]",
            "true",
            // An object with a content_type but no marker is still a document:
            // only the marker decides.
            r#"{"content_type":"text/plain"}"#,
        ] {
            let parsed: StoredVal = serde_json::from_str(text).unwrap();
            assert!(
                parsed.as_json().is_some(),
                "`{text}` must load as JSON, not as raw bytes"
            );
        }
    }

    #[test]
    fn raw_bytes_roundtrip_with_content_type() {
        let value = StoredVal::raw((0u8..=255).collect(), Some("image/png".into()));
        let back = roundtrip(&value);
        assert_eq!(back, value);
        assert_eq!(back.as_bytes().unwrap().len(), 256);
        assert_eq!(back.content_type(), Some("image/png"));
    }

    #[test]
    fn raw_bytes_roundtrip_without_content_type() {
        let value = StoredVal::raw(vec![0, 159, 146, 150], None);
        let back = roundtrip(&value);
        assert_eq!(back, value);
        assert_eq!(back.content_type(), None);
    }

    #[test]
    fn empty_and_large_payloads_survive() {
        assert_eq!(
            roundtrip(&StoredVal::raw(vec![], None)).as_bytes(),
            Some(&[][..])
        );
        let big = StoredVal::raw(vec![0xAB; 1024 * 1024], None);
        assert_eq!(roundtrip(&big).as_bytes().unwrap().len(), 1024 * 1024);
    }

    #[test]
    fn bytes_that_are_not_valid_utf8_survive() {
        // The whole reason this type exists: these bytes cannot be a JSON
        // string, so the old Value-only store could not hold them at all.
        let value = StoredVal::raw(vec![0xFF, 0xFE, 0x00, 0x80], None);
        assert_eq!(roundtrip(&value), value);
    }

    #[test]
    fn ambiguous_json_is_refused_at_the_boundary() {
        let ambiguous = json!({ RAW_MARKER: "not really base64 bytes" });
        assert!(
            StoredVal::from_json_checked(ambiguous).is_err(),
            "a document carrying the reserved marker must be refused at write \
             time rather than changing type on the way back out"
        );

        // Nested is fine: only the top level is inspected when decoding.
        let nested = json!({ "payload": { RAW_MARKER: "abc" } });
        assert!(StoredVal::from_json_checked(nested).is_ok());
    }

    #[test]
    fn a_malformed_marker_is_an_error_not_a_silent_json_value() {
        // If this fell back to Json, a corrupted record would come back as a
        // document with a base64-looking field and the caller would never know
        // its bytes were gone.
        let err = serde_json::from_str::<StoredVal>(r#"{"__luma_raw":"!!!not base64!!!"}"#);
        assert!(err.is_err(), "a broken marker must fail loudly");

        let err = serde_json::from_str::<StoredVal>(r#"{"__luma_raw":123}"#);
        assert!(err.is_err(), "a non-string marker must fail loudly");
    }

    #[test]
    fn size_hint_measures_what_is_stored() {
        assert_eq!(StoredVal::raw(vec![1, 2, 3], None).size_hint(), 3);
        assert_eq!(StoredVal::Json(json!(12345)).size_hint(), 5);
    }

    #[test]
    fn as_json_is_none_for_raw_so_indexing_skips_it() {
        // The secondary index walks JSON fields; a raw value simply has none,
        // which keeps it out of the index without a special case there.
        assert!(StoredVal::raw(vec![1], None).as_json().is_none());
    }
}
