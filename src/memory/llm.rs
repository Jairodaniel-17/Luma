use anyhow::{anyhow, Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FactCandidate {
    pub fact_key: String,
    pub content: String,
    pub confidence: f32,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[derive(Clone, Debug)]
pub enum InferenceProvider {
    OpenAI {
        api_url: String,
        api_key: String,
        model: String,
    },
    Ollama {
        api_url: String,
        model: String,
    },
    None,
    Mock,
}

#[derive(Clone, Debug)]
pub struct InferenceClient {
    provider: InferenceProvider,
    client: Client,
}

impl Default for InferenceClient {
    fn default() -> Self {
        Self {
            provider: InferenceProvider::None,
            client: Client::new(),
        }
    }
}

impl InferenceClient {
    pub fn new(provider: InferenceProvider) -> Self {
        Self {
            provider,
            client: Client::new(),
        }
    }

    pub async fn extract_facts(
        &self,
        text: &str,
        metadata: &serde_json::Value,
    ) -> Result<Vec<FactCandidate>> {
        let heuristic = heuristic_extract_facts(text, metadata);
        let provider_facts = match &self.provider {
            InferenceProvider::None => Vec::new(),
            InferenceProvider::Mock => heuristic.clone(),
            InferenceProvider::OpenAI {
                api_url,
                api_key,
                model,
            } => self
                .extract_openai(api_url, api_key, model, text, metadata)
                .await
                .unwrap_or_default(),
            InferenceProvider::Ollama { api_url, model } => self
                .extract_ollama(api_url, model, text, metadata)
                .await
                .unwrap_or_default(),
        };

        Ok(merge_candidates(provider_facts, heuristic))
    }

    async fn extract_openai(
        &self,
        api_url: &str,
        api_key: &str,
        model: &str,
        text: &str,
        metadata: &serde_json::Value,
    ) -> Result<Vec<FactCandidate>> {
        #[derive(Serialize)]
        struct Message<'a> {
            role: &'a str,
            content: &'a str,
        }

        #[derive(Serialize)]
        struct Req<'a> {
            model: &'a str,
            response_format: serde_json::Value,
            messages: Vec<Message<'a>>,
        }

        #[derive(Deserialize)]
        struct Resp {
            choices: Vec<Choice>,
        }

        #[derive(Deserialize)]
        struct Choice {
            message: ChoiceMessage,
        }

        #[derive(Deserialize)]
        struct ChoiceMessage {
            content: String,
        }

        let prompt = build_prompt(text, metadata);
        let resp = self
            .client
            .post(api_url)
            .bearer_auth(api_key)
            .json(&Req {
                model,
                response_format: serde_json::json!({ "type": "json_object" }),
                messages: vec![
                    Message {
                        role: "system",
                        content: SYSTEM_PROMPT,
                    },
                    Message {
                        role: "user",
                        content: &prompt,
                    },
                ],
            })
            .send()
            .await
            .context("openai fact extraction request")?;

        if !resp.status().is_success() {
            return Err(anyhow!(
                "OpenAI API error: {}",
                resp.text().await.unwrap_or_default()
            ));
        }

        let parsed: Resp = resp.json().await.context("parse openai fact extraction")?;
        let content = parsed
            .choices
            .first()
            .map(|choice| choice.message.content.clone())
            .ok_or_else(|| anyhow!("no choices returned from openai fact extraction"))?;
        parse_candidates_payload(&content)
    }

    async fn extract_ollama(
        &self,
        api_url: &str,
        model: &str,
        text: &str,
        metadata: &serde_json::Value,
    ) -> Result<Vec<FactCandidate>> {
        #[derive(Serialize)]
        struct Req<'a> {
            model: &'a str,
            prompt: String,
            stream: bool,
        }

        #[derive(Deserialize)]
        struct Resp {
            response: String,
        }

        let resp = self
            .client
            .post(api_url)
            .json(&Req {
                model,
                prompt: format!("{SYSTEM_PROMPT}\n\n{}", build_prompt(text, metadata)),
                stream: false,
            })
            .send()
            .await
            .context("ollama fact extraction request")?;

        if !resp.status().is_success() {
            return Err(anyhow!(
                "Ollama API error: {}",
                resp.text().await.unwrap_or_default()
            ));
        }

        let parsed: Resp = resp.json().await.context("parse ollama fact extraction")?;
        parse_candidates_payload(&parsed.response)
    }
}

const SYSTEM_PROMPT: &str = "Extract stable semantic facts from the event. Return strict JSON with shape {\"facts\":[{\"fact_key\":\"...\",\"content\":\"...\",\"confidence\":0.0,\"metadata\":{}}]}. Only include durable facts, preferences, profile information, or persistent business state.";

fn build_prompt(text: &str, metadata: &serde_json::Value) -> String {
    format!(
        "Event text:\n{text}\n\nEvent metadata:\n{}\n\nReturn only JSON.",
        metadata
    )
}

fn parse_candidates_payload(payload: &str) -> Result<Vec<FactCandidate>> {
    #[derive(Deserialize)]
    struct Envelope {
        #[serde(default)]
        facts: Vec<FactCandidate>,
    }

    if let Ok(envelope) = serde_json::from_str::<Envelope>(payload) {
        return Ok(normalize_candidates(envelope.facts));
    }
    if let Ok(candidates) = serde_json::from_str::<Vec<FactCandidate>>(payload) {
        return Ok(normalize_candidates(candidates));
    }
    Err(anyhow!("invalid fact extraction payload"))
}

fn normalize_candidates(candidates: Vec<FactCandidate>) -> Vec<FactCandidate> {
    candidates
        .into_iter()
        .filter(|candidate| {
            !candidate.fact_key.trim().is_empty() && !candidate.content.trim().is_empty()
        })
        .map(|mut candidate| {
            candidate.confidence = candidate.confidence.clamp(0.0, 1.0);
            candidate
        })
        .collect()
}

fn merge_candidates(
    primary: Vec<FactCandidate>,
    fallback: Vec<FactCandidate>,
) -> Vec<FactCandidate> {
    let mut out = primary;
    for candidate in fallback {
        if out
            .iter()
            .any(|existing| existing.fact_key == candidate.fact_key)
        {
            continue;
        }
        out.push(candidate);
    }
    out
}

pub fn heuristic_extract_facts(text: &str, metadata: &serde_json::Value) -> Vec<FactCandidate> {
    let mut out = Vec::new();
    let lower = text.to_ascii_lowercase();

    if lower.contains("alertas por correo")
        || lower.contains("email alerts")
        || lower.contains("correo electrónico")
    {
        out.push(FactCandidate {
            fact_key: "notification_preference".to_string(),
            content: "Prefiere alertas por correo".to_string(),
            confidence: 0.92,
            metadata: serde_json::json!({ "derived_from": "heuristic", "channel": metadata.get("channel") }),
        });
    }

    if let Some(preference) =
        extract_after_any(text, &["prefiere ", "prefers ", "wants ", "quiere "])
    {
        out.push(FactCandidate {
            fact_key: "user_preference".to_string(),
            content: format!("Prefiere {}", preference.trim()),
            confidence: 0.88,
            metadata: serde_json::json!({ "derived_from": "heuristic" }),
        });
    }

    if lower.contains("pidió ")
        || lower.contains("solicitó ")
        || lower.contains("requested ")
        || lower.contains("asked for ")
    {
        out.push(FactCandidate {
            fact_key: "recent_request".to_string(),
            content: text.trim().to_string(),
            confidence: 0.81,
            metadata: serde_json::json!({ "derived_from": "heuristic" }),
        });
    }

    out
}

fn extract_after_any<'a>(text: &'a str, needles: &[&str]) -> Option<&'a str> {
    let lower = text.to_ascii_lowercase();
    for needle in needles {
        if let Some(idx) = lower.find(needle) {
            let start = idx + needle.len();
            let tail = &text[start..];
            let end = tail.find(['.', ',', ';', '\n']).unwrap_or(tail.len());
            let value = tail[..end].trim();
            if !value.is_empty() {
                return Some(value);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::heuristic_extract_facts;

    #[test]
    fn heuristic_detects_email_preferences() {
        let facts = heuristic_extract_facts(
            "El usuario pidió activar alertas por correo",
            &serde_json::json!({ "channel": "chat" }),
        );
        assert!(facts
            .iter()
            .any(|fact| fact.fact_key == "notification_preference"));
    }
}
