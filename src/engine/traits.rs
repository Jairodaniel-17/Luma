use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;

#[async_trait]
pub trait BaseEngine: Send + Sync {
    async fn add(&self, collection: &str, id: &str, data: Value) -> Result<()>;
    async fn search(&self, collection: &str, query: Value) -> Result<Vec<Value>>;
    async fn delete(&self, collection: &str, id: &str) -> Result<bool>;
    async fn health(&self) -> bool;
    async fn stats(&self) -> Value;
}
