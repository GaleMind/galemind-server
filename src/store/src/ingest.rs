//! Moving model artifacts from one store into another.

use galemind_core::{InferError, ModelKey, ModelStore, ModelWriter};
use tracing::{debug, info, warn};

/// What an ingest run did.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct IngestReport {
    /// Artifacts newly written to the target.
    pub added: usize,
    /// Artifacts replaced because their bytes changed.
    pub updated: usize,
    /// Artifacts skipped because the target already held identical bytes.
    pub unchanged: usize,
    /// Artifacts that could not be copied.
    pub failed: usize,
}

impl IngestReport {
    pub fn total(&self) -> usize {
        self.added + self.updated + self.unchanged + self.failed
    }
}

/// Copies every model in `source` into `target`, skipping artifacts whose
/// digest already matches so that a restart does not rewrite the whole
/// repository.
///
/// A model that fails to copy is logged and counted, not fatal: one unreadable
/// artifact must not stop the server from serving the others.
pub async fn ingest<S, T>(source: &S, target: &T) -> Result<IngestReport, InferError>
where
    S: ModelStore + ?Sized,
    T: ModelStore + ModelWriter + ?Sized,
{
    let keys = source.list().await?;
    let mut report = IngestReport::default();

    for key in keys {
        match ingest_one(source, target, &key).await {
            Ok(Outcome::Added) => report.added += 1,
            Ok(Outcome::Updated) => report.updated += 1,
            Ok(Outcome::Unchanged) => report.unchanged += 1,
            Err(e) => {
                report.failed += 1;
                warn!(model = %key, error = %e, "skipping model that could not be ingested");
            }
        }
    }

    info!(
        added = report.added,
        updated = report.updated,
        unchanged = report.unchanged,
        failed = report.failed,
        "ingested model repository"
    );
    Ok(report)
}

enum Outcome {
    Added,
    Updated,
    Unchanged,
}

async fn ingest_one<S, T>(source: &S, target: &T, key: &ModelKey) -> Result<Outcome, InferError>
where
    S: ModelStore + ?Sized,
    T: ModelStore + ModelWriter + ?Sized,
{
    let artifact = source.fetch(key).await?;
    let existing = target.digest(key).await?;

    match existing {
        Some(digest) if digest == artifact.digest => {
            debug!(model = %key, "artifact unchanged");
            Ok(Outcome::Unchanged)
        }
        existing => {
            target.put(&artifact).await?;
            Ok(if existing.is_some() {
                Outcome::Updated
            } else {
                Outcome::Added
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::digest::sha256_hex;
    use crate::memory::MemoryModelStore;
    use galemind_core::{ModelArtifact, ModelConfig, PLATFORM_ONNX};

    fn artifact(name: &str, bytes: &[u8]) -> ModelArtifact {
        ModelArtifact {
            key: ModelKey::new(name, "1"),
            platform: PLATFORM_ONNX.to_string(),
            digest: sha256_hex(bytes),
            config: ModelConfig::default(),
            bytes: bytes.to_vec(),
        }
    }

    #[tokio::test]
    async fn a_first_run_adds_everything_and_a_second_run_adds_nothing() {
        let source = MemoryModelStore::new();
        let target = MemoryModelStore::new();
        source.put(&artifact("energy", b"a")).await.unwrap();
        source.put(&artifact("bands", b"b")).await.unwrap();

        let first = ingest(&source, &target).await.unwrap();
        assert_eq!(
            first,
            IngestReport {
                added: 2,
                ..Default::default()
            }
        );

        let second = ingest(&source, &target).await.unwrap();
        assert_eq!(
            second,
            IngestReport {
                unchanged: 2,
                ..Default::default()
            }
        );
        assert_eq!(second.total(), 2);
    }

    #[tokio::test]
    async fn changed_bytes_are_copied_again() {
        let source = MemoryModelStore::new();
        let target = MemoryModelStore::new();
        source.put(&artifact("energy", b"old")).await.unwrap();
        ingest(&source, &target).await.unwrap();

        source.put(&artifact("energy", b"new")).await.unwrap();
        let report = ingest(&source, &target).await.unwrap();
        assert_eq!(report.updated, 1);
        assert_eq!(
            target
                .fetch(&ModelKey::new("energy", "1"))
                .await
                .unwrap()
                .bytes,
            b"new"
        );
    }
}
