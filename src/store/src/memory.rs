//! In-memory stores, used for tests and as the fallback when the RocksDB
//! feature is disabled.

use async_trait::async_trait;
use galemind_core::{
    InferError, ModelArtifact, ModelKey, ModelStore, ModelUsage, ModelWriter, UsageIndex,
};
use std::collections::HashMap;
use std::sync::Mutex;

type Result<T> = std::result::Result<T, InferError>;

/// Holds artifacts and usage records for the lifetime of the process.
#[derive(Default)]
pub struct MemoryModelStore {
    artifacts: Mutex<HashMap<ModelKey, ModelArtifact>>,
    usage: Mutex<HashMap<ModelKey, ModelUsage>>,
}

impl MemoryModelStore {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock<T>(guard: std::sync::LockResult<T>) -> Result<T> {
        guard.map_err(|_| InferError::Internal("in-memory store lock poisoned".to_string()))
    }
}

#[async_trait]
impl ModelStore for MemoryModelStore {
    async fn list(&self) -> Result<Vec<ModelKey>> {
        let artifacts = Self::lock(self.artifacts.lock())?;
        let mut keys: Vec<_> = artifacts.keys().cloned().collect();
        keys.sort();
        Ok(keys)
    }

    async fn fetch(&self, key: &ModelKey) -> Result<ModelArtifact> {
        Self::lock(self.artifacts.lock())?
            .get(key)
            .cloned()
            .ok_or_else(|| InferError::VersionNotFound {
                name: key.name.clone(),
                version: key.version.clone(),
            })
    }
}

#[async_trait]
impl ModelWriter for MemoryModelStore {
    async fn put(&self, artifact: &ModelArtifact) -> Result<()> {
        Self::lock(self.artifacts.lock())?.insert(artifact.key.clone(), artifact.clone());
        Ok(())
    }

    async fn digest(&self, key: &ModelKey) -> Result<Option<String>> {
        Ok(Self::lock(self.artifacts.lock())?
            .get(key)
            .map(|a| a.digest.clone()))
    }
}

#[async_trait]
impl UsageIndex for MemoryModelStore {
    async fn touch(&self, key: &ModelKey, at_unix_ms: u64, size_bytes: usize) -> Result<()> {
        let mut usage = Self::lock(self.usage.lock())?;
        let record = usage.entry(key.clone()).or_insert_with(|| ModelUsage {
            key: key.clone(),
            last_used_unix_ms: 0,
            load_count: 0,
            size_bytes: 0,
        });
        record.last_used_unix_ms = at_unix_ms;
        record.load_count += 1;
        record.size_bytes = size_bytes;
        Ok(())
    }

    async fn usage(&self) -> Result<Vec<ModelUsage>> {
        let usage = Self::lock(self.usage.lock())?;
        let mut records: Vec<_> = usage.values().cloned().collect();
        records.sort_by(|a, b| b.last_used_unix_ms.cmp(&a.last_used_unix_ms));
        Ok(records)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use galemind_core::{ModelConfig, PLATFORM_ONNX};

    fn artifact(name: &str) -> ModelArtifact {
        ModelArtifact {
            key: ModelKey::new(name, "1"),
            platform: PLATFORM_ONNX.to_string(),
            digest: "digest".to_string(),
            config: ModelConfig::default(),
            bytes: b"bytes".to_vec(),
        }
    }

    #[tokio::test]
    async fn artifacts_round_trip_and_unknown_keys_are_reported() {
        let store = MemoryModelStore::new();
        store.put(&artifact("energy")).await.unwrap();

        assert_eq!(
            store.list().await.unwrap(),
            vec![ModelKey::new("energy", "1")]
        );
        assert_eq!(
            store
                .fetch(&ModelKey::new("energy", "1"))
                .await
                .unwrap()
                .bytes,
            b"bytes"
        );
        assert!(store.fetch(&ModelKey::new("ghost", "1")).await.is_err());
        assert_eq!(
            store.digest(&ModelKey::new("energy", "1")).await.unwrap(),
            Some("digest".to_string())
        );
    }

    #[tokio::test]
    async fn usage_records_accumulate_load_counts() {
        let store = MemoryModelStore::new();
        let key = ModelKey::new("energy", "1");
        store.touch(&key, 10, 4).await.unwrap();
        store.touch(&key, 20, 4).await.unwrap();

        let usage = store.usage().await.unwrap();
        assert_eq!(usage[0].load_count, 2);
        assert_eq!(usage[0].last_used_unix_ms, 20);
    }
}
