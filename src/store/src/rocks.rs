//! The RocksDB-backed model store.
//!
//! RocksDB is the cold tier of the two-tier model cache. A loaded model is a
//! live runtime session that cannot be serialized, so what is persisted here is
//! everything needed to rebuild one quickly:
//!
//! | column family | key             | value                                  |
//! |---------------|-----------------|----------------------------------------|
//! | `artifacts`   | `name/version`  | the model file's bytes                 |
//! | `metadata`    | `name/version`  | platform, digest and model config      |
//! | `usage`       | `name/version`  | last used, load count, resident size   |
//!
//! Persisting usage is what lets the in-memory cache keep a sensible eviction
//! order across a restart instead of starting cold and blind.

use async_trait::async_trait;
use galemind_core::{
    InferError, ModelArtifact, ModelConfig, ModelKey, ModelStore, ModelUsage, ModelWriter,
    UsageIndex,
};
use rocksdb::{ColumnFamilyDescriptor, IteratorMode, Options, DB};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use tracing::debug;

type Result<T> = std::result::Result<T, InferError>;

const CF_ARTIFACTS: &str = "artifacts";
const CF_METADATA: &str = "metadata";
const CF_USAGE: &str = "usage";

/// What the `metadata` column family holds for each model version.
#[derive(Debug, Serialize, Deserialize)]
struct StoredMetadata {
    platform: String,
    digest: String,
    #[serde(default)]
    max_batch_size: Option<usize>,
    #[serde(default)]
    intra_op_threads: Option<usize>,
}

/// What the `usage` column family holds for each model version.
#[derive(Debug, Default, Serialize, Deserialize)]
struct StoredUsage {
    last_used_unix_ms: u64,
    load_count: u64,
    size_bytes: usize,
}

/// A model store persisted in RocksDB.
pub struct RocksModelStore {
    db: Arc<DB>,
}

impl RocksModelStore {
    /// Opens, creating the database and its column families if needed.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let mut options = Options::default();
        options.create_if_missing(true);
        options.create_missing_column_families(true);
        options.set_compression_type(rocksdb::DBCompressionType::Lz4);

        let families = [CF_ARTIFACTS, CF_METADATA, CF_USAGE]
            .into_iter()
            .map(|name| ColumnFamilyDescriptor::new(name, Options::default()))
            .collect::<Vec<_>>();

        let db = DB::open_cf_descriptors(&options, path.as_ref(), families).map_err(|e| {
            InferError::Store(format!(
                "cannot open model store at {}: {e}",
                path.as_ref().display()
            ))
        })?;

        Ok(Self { db: Arc::new(db) })
    }

    fn family<'a>(db: &'a DB, name: &str) -> Result<&'a rocksdb::ColumnFamily> {
        db.cf_handle(name)
            .ok_or_else(|| InferError::Store(format!("column family '{name}' is missing")))
    }

    /// Runs a blocking RocksDB operation off the async runtime.
    async fn with_db<T, F>(&self, operation: F) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce(&DB) -> Result<T> + Send + 'static,
    {
        let db = Arc::clone(&self.db);
        tokio::task::spawn_blocking(move || operation(&db))
            .await
            .map_err(|e| InferError::Internal(format!("model store task failed: {e}")))?
    }

    fn read_metadata(db: &DB, key: &ModelKey) -> Result<StoredMetadata> {
        let family = Self::family(db, CF_METADATA)?;
        let raw = db
            .get_cf(family, key.storage_key())
            .map_err(|e| InferError::Store(format!("cannot read metadata for {key}: {e}")))?
            .ok_or_else(|| InferError::VersionNotFound {
                name: key.name.clone(),
                version: key.version.clone(),
            })?;
        serde_json::from_slice(&raw)
            .map_err(|e| InferError::Store(format!("corrupt metadata for {key}: {e}")))
    }
}

#[async_trait]
impl ModelStore for RocksModelStore {
    async fn list(&self) -> Result<Vec<ModelKey>> {
        self.with_db(|db| {
            let family = Self::family(db, CF_METADATA)?;
            let mut keys = Vec::new();
            for item in db.iterator_cf(family, IteratorMode::Start) {
                let (raw, _) =
                    item.map_err(|e| InferError::Store(format!("cannot list models: {e}")))?;
                let text = std::str::from_utf8(&raw)
                    .map_err(|e| InferError::Store(format!("non-UTF-8 model key: {e}")))?;
                keys.push(ModelKey::from_storage_key(text)?);
            }
            keys.sort();
            Ok(keys)
        })
        .await
    }

    async fn fetch(&self, key: &ModelKey) -> Result<ModelArtifact> {
        let key = key.clone();
        self.with_db(move |db| {
            let metadata = Self::read_metadata(db, &key)?;
            let artifacts = Self::family(db, CF_ARTIFACTS)?;
            let bytes = db
                .get_cf(artifacts, key.storage_key())
                .map_err(|e| InferError::Store(format!("cannot read artifact {key}: {e}")))?
                .ok_or_else(|| {
                    InferError::Store(format!("metadata for {key} has no stored artifact"))
                })?;

            Ok(ModelArtifact {
                key,
                platform: metadata.platform,
                digest: metadata.digest,
                config: ModelConfig {
                    platform: None,
                    max_batch_size: metadata.max_batch_size,
                    intra_op_threads: metadata.intra_op_threads,
                },
                bytes,
            })
        })
        .await
    }
}

#[async_trait]
impl ModelWriter for RocksModelStore {
    async fn put(&self, artifact: &ModelArtifact) -> Result<()> {
        let key = artifact.key.clone();
        let bytes = artifact.bytes.clone();
        let metadata = StoredMetadata {
            platform: artifact.platform.clone(),
            digest: artifact.digest.clone(),
            max_batch_size: artifact.config.max_batch_size,
            intra_op_threads: artifact.config.intra_op_threads,
        };

        self.with_db(move |db| {
            let encoded = serde_json::to_vec(&metadata)
                .map_err(|e| InferError::Store(format!("cannot encode metadata: {e}")))?;
            let storage_key = key.storage_key();

            let mut batch = rocksdb::WriteBatch::default();
            batch.put_cf(Self::family(db, CF_ARTIFACTS)?, &storage_key, &bytes);
            batch.put_cf(Self::family(db, CF_METADATA)?, &storage_key, &encoded);
            db.write(batch)
                .map_err(|e| InferError::Store(format!("cannot store {key}: {e}")))?;

            debug!(model = %key, bytes = bytes.len(), "stored model artifact");
            Ok(())
        })
        .await
    }

    async fn digest(&self, key: &ModelKey) -> Result<Option<String>> {
        let key = key.clone();
        self.with_db(move |db| match Self::read_metadata(db, &key) {
            Ok(metadata) => Ok(Some(metadata.digest)),
            Err(InferError::VersionNotFound { .. }) => Ok(None),
            Err(e) => Err(e),
        })
        .await
    }
}

#[async_trait]
impl UsageIndex for RocksModelStore {
    async fn touch(&self, key: &ModelKey, at_unix_ms: u64, size_bytes: usize) -> Result<()> {
        let key = key.clone();
        self.with_db(move |db| {
            let family = Self::family(db, CF_USAGE)?;
            let storage_key = key.storage_key();

            let mut usage: StoredUsage = db
                .get_cf(family, &storage_key)
                .map_err(|e| InferError::Store(format!("cannot read usage for {key}: {e}")))?
                .and_then(|raw| serde_json::from_slice(&raw).ok())
                .unwrap_or_default();

            usage.last_used_unix_ms = at_unix_ms;
            usage.load_count += 1;
            usage.size_bytes = size_bytes;

            let encoded = serde_json::to_vec(&usage)
                .map_err(|e| InferError::Store(format!("cannot encode usage: {e}")))?;
            db.put_cf(family, &storage_key, encoded)
                .map_err(|e| InferError::Store(format!("cannot record usage for {key}: {e}")))
        })
        .await
    }

    async fn usage(&self) -> Result<Vec<ModelUsage>> {
        self.with_db(|db| {
            let family = Self::family(db, CF_USAGE)?;
            let mut records = Vec::new();
            for item in db.iterator_cf(family, IteratorMode::Start) {
                let (raw_key, raw_value) =
                    item.map_err(|e| InferError::Store(format!("cannot read usage: {e}")))?;
                let text = std::str::from_utf8(&raw_key)
                    .map_err(|e| InferError::Store(format!("non-UTF-8 usage key: {e}")))?;
                let key = ModelKey::from_storage_key(text)?;
                let usage: StoredUsage = serde_json::from_slice(&raw_value)
                    .map_err(|e| InferError::Store(format!("corrupt usage for {key}: {e}")))?;
                records.push(ModelUsage {
                    key,
                    last_used_unix_ms: usage.last_used_unix_ms,
                    load_count: usage.load_count,
                    size_bytes: usage.size_bytes,
                });
            }
            records.sort_by(|a, b| b.last_used_unix_ms.cmp(&a.last_used_unix_ms));
            Ok(records)
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::digest::sha256_hex;
    use galemind_core::PLATFORM_ONNX;

    fn artifact(name: &str, version: &str, bytes: &[u8]) -> ModelArtifact {
        ModelArtifact {
            key: ModelKey::new(name, version),
            platform: PLATFORM_ONNX.to_string(),
            digest: sha256_hex(bytes),
            config: ModelConfig {
                platform: None,
                max_batch_size: Some(8),
                intra_op_threads: Some(2),
            },
            bytes: bytes.to_vec(),
        }
    }

    async fn store() -> (tempfile::TempDir, RocksModelStore) {
        let dir = tempfile::tempdir().unwrap();
        let store = RocksModelStore::open(dir.path().join("db")).unwrap();
        (dir, store)
    }

    #[tokio::test]
    async fn a_stored_artifact_comes_back_unchanged() {
        let (_dir, store) = store().await;
        let original = artifact("energy", "1", b"onnx-bytes");
        store.put(&original).await.unwrap();

        let fetched = store.fetch(&original.key).await.unwrap();
        assert_eq!(fetched.bytes, original.bytes);
        assert_eq!(fetched.platform, PLATFORM_ONNX);
        assert_eq!(fetched.digest, original.digest);
        assert_eq!(fetched.config.max_batch_size, Some(8));
        assert_eq!(fetched.config.intra_op_threads, Some(2));
    }

    #[tokio::test]
    async fn listing_returns_every_stored_version_in_order() {
        let (_dir, store) = store().await;
        store.put(&artifact("energy", "2", b"b")).await.unwrap();
        store.put(&artifact("energy", "1", b"a")).await.unwrap();
        store.put(&artifact("bands", "1", b"c")).await.unwrap();

        assert_eq!(
            store.list().await.unwrap(),
            vec![
                ModelKey::new("bands", "1"),
                ModelKey::new("energy", "1"),
                ModelKey::new("energy", "2"),
            ]
        );
    }

    #[tokio::test]
    async fn the_digest_of_an_unknown_model_is_none() {
        let (_dir, store) = store().await;
        assert_eq!(
            store.digest(&ModelKey::new("ghost", "1")).await.unwrap(),
            None
        );
    }

    #[tokio::test]
    async fn putting_the_same_version_twice_replaces_it() {
        let (_dir, store) = store().await;
        store.put(&artifact("energy", "1", b"old")).await.unwrap();
        store.put(&artifact("energy", "1", b"new")).await.unwrap();

        assert_eq!(store.list().await.unwrap().len(), 1);
        let fetched = store.fetch(&ModelKey::new("energy", "1")).await.unwrap();
        assert_eq!(fetched.bytes, b"new");
        assert_eq!(fetched.digest, sha256_hex(b"new"));
    }

    #[tokio::test]
    async fn fetching_an_unknown_model_reports_version_not_found() {
        let (_dir, store) = store().await;
        let err = store.fetch(&ModelKey::new("ghost", "1")).await.unwrap_err();
        assert!(matches!(err, InferError::VersionNotFound { .. }), "{err}");
    }

    #[tokio::test]
    async fn usage_accumulates_and_is_reported_newest_first() {
        let (_dir, store) = store().await;
        let energy = ModelKey::new("energy", "1");
        let bands = ModelKey::new("bands", "1");

        store.touch(&energy, 1_000, 512).await.unwrap();
        store.touch(&energy, 2_000, 512).await.unwrap();
        store.touch(&bands, 3_000, 256).await.unwrap();

        let usage = store.usage().await.unwrap();
        assert_eq!(usage.len(), 2);
        assert_eq!(usage[0].key, bands);
        assert_eq!(usage[0].last_used_unix_ms, 3_000);
        assert_eq!(usage[1].key, energy);
        assert_eq!(usage[1].load_count, 2);
        assert_eq!(usage[1].size_bytes, 512);
    }

    #[tokio::test]
    async fn stored_state_survives_reopening_the_database() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db");
        {
            let store = RocksModelStore::open(&path).unwrap();
            store.put(&artifact("energy", "1", b"bytes")).await.unwrap();
            store
                .touch(&ModelKey::new("energy", "1"), 9_000, 5)
                .await
                .unwrap();
        }
        let reopened = RocksModelStore::open(&path).unwrap();
        assert_eq!(reopened.list().await.unwrap().len(), 1);
        assert_eq!(reopened.usage().await.unwrap()[0].last_used_unix_ms, 9_000);
    }
}
