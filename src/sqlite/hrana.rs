/// Hrana v2 over HTTP — Turso / self-hosted libSQL remote backend.
///
/// Hrana is the wire protocol used by libSQL/Turso for remote SQL over HTTPS.
/// Implementing it directly via reqwest avoids bundling a second SQLite fork.
///
/// Spec: https://github.com/tursodatabase/libsql/blob/main/docs/HRANA_3_SPEC.md
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// ---- Request types ----

#[derive(Serialize)]
struct PipelineReq<'a> {
    baton: Option<&'a str>,
    requests: Vec<HranaReq<'a>>,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum HranaReq<'a> {
    Execute { stmt: HranaStmt<'a> },
    Close,
}

#[derive(Serialize)]
struct HranaStmt<'a> {
    sql: &'a str,
    args: Vec<HranaArg>,
    want_rows: bool,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum HranaArg {
    Null,
    Integer { value: String }, // integer value is string-encoded in Hrana
    Float { value: f64 },
    Text { value: String },
    // Blob not included: serde_json::Value has no binary type, so it can never be constructed
}

// ---- Response types ----

#[derive(Deserialize)]
struct PipelineResp {
    results: Vec<HranaResult>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum HranaResult {
    Ok { response: HranaResp },
    Error { error: HranaErr },
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum HranaResp {
    Execute { result: ExecResult },
    Close,
}

#[derive(Deserialize)]
struct ExecResult {
    cols: Vec<HranaCol>,
    rows: Vec<Vec<HranaVal>>,
    affected_row_count: u64,
}

#[derive(Deserialize)]
struct HranaCol {
    name: String,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum HranaVal {
    Null,
    Integer { value: String },
    Float { value: f64 },
    Text { value: String },
    Blob { base64: String },
}

#[derive(Deserialize)]
struct HranaErr {
    message: String,
}

// ---- Service ----

#[derive(Clone)]
pub struct HranaService {
    inner: Arc<HranaInner>,
}

struct HranaInner {
    client: Client,
    url: String, // e.g. "https://db-name.turso.io"
    token: String,
}

impl HranaService {
    pub fn new(url: String, token: String) -> Self {
        Self {
            inner: Arc::new(HranaInner {
                client: Client::new(),
                url,
                token,
            }),
        }
    }

    pub async fn execute(
        &self,
        sql: String,
        params: Vec<serde_json::Value>,
    ) -> anyhow::Result<u64> {
        let args = to_hrana_args(params)?;
        let stmt = HranaStmt {
            sql: &sql,
            args,
            want_rows: false,
        };
        let result = self.pipeline(stmt).await?;
        Ok(result.affected_row_count)
    }

    pub async fn query(
        &self,
        sql: String,
        params: Vec<serde_json::Value>,
    ) -> anyhow::Result<Vec<serde_json::Value>> {
        let args = to_hrana_args(params)?;
        let stmt = HranaStmt {
            sql: &sql,
            args,
            want_rows: true,
        };
        let result = self.pipeline(stmt).await?;
        let col_names: Vec<&str> = result.cols.iter().map(|c| c.name.as_str()).collect();
        let mut rows = Vec::with_capacity(result.rows.len());
        for row in result.rows {
            let mut obj = serde_json::Map::new();
            for (col, val) in col_names.iter().zip(row) {
                obj.insert((*col).to_string(), from_hrana_val(val));
            }
            rows.push(serde_json::Value::Object(obj));
        }
        Ok(rows)
    }

    async fn pipeline(&self, stmt: HranaStmt<'_>) -> anyhow::Result<ExecResult> {
        let body = PipelineReq {
            baton: None,
            requests: vec![HranaReq::Execute { stmt }, HranaReq::Close],
        };
        let resp = self
            .inner
            .client
            .post(format!("{}/v2/pipeline", self.inner.url))
            .header("Authorization", format!("Bearer {}", self.inner.token))
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json::<PipelineResp>()
            .await?;

        match resp.results.into_iter().next() {
            Some(HranaResult::Ok {
                response: HranaResp::Execute { result },
            }) => Ok(result),
            Some(HranaResult::Error { error }) => {
                anyhow::bail!("libSQL remote error: {}", error.message)
            }
            _ => anyhow::bail!("unexpected libSQL pipeline response"),
        }
    }
}

fn to_hrana_args(params: Vec<serde_json::Value>) -> anyhow::Result<Vec<HranaArg>> {
    params
        .into_iter()
        .map(|v| {
            Ok(match v {
                serde_json::Value::Null => HranaArg::Null,
                serde_json::Value::Bool(b) => HranaArg::Integer {
                    value: if b { "1".into() } else { "0".into() },
                },
                serde_json::Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        HranaArg::Integer {
                            value: i.to_string(),
                        }
                    } else if let Some(f) = n.as_f64() {
                        HranaArg::Float { value: f }
                    } else {
                        HranaArg::Null
                    }
                }
                serde_json::Value::String(s) => HranaArg::Text { value: s },
                _ => {
                    anyhow::bail!("unsupported Hrana parameter type (only null/bool/number/string)")
                }
            })
        })
        .collect()
}

fn from_hrana_val(val: HranaVal) -> serde_json::Value {
    match val {
        HranaVal::Null => serde_json::Value::Null,
        HranaVal::Integer { value } => value
            .parse::<i64>()
            .ok()
            .map(|i| serde_json::json!(i))
            .unwrap_or(serde_json::Value::String(value)),
        HranaVal::Float { value } => serde_json::Number::from_f64(value)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        HranaVal::Text { value } => serde_json::Value::String(value),
        HranaVal::Blob { base64 } => serde_json::Value::String(base64),
    }
}
