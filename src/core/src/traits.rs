//! The seams of the server.
//!
//! Every replaceable part sits behind one of these traits, so the transport
//! layers and the orchestration layer depend on behaviour rather than on a
//! concrete runtime, database or framework.

use crate::error::InferError;
use crate::infer::{InferRequest, InferResponse};
use crate::model::{ModelArtifact, ModelKey, ModelMetadata};
use async_trait::async_trait;
use std::sync::Arc;

type Result<T> = std::result::Result<T, InferError>;

/// One loaded model, ready to execute.
///
/// `infer` is synchronous and may block: inference is CPU bound, so callers run
/// it on a blocking thread rather than on an async worker.
pub trait Backend: Send + Sync {
    /// Shapes and types this model accepts and produces, as reported by the
    /// runtime that loaded it.
    fn metadata(&self) -> &ModelMetadata;

    /// Executes one request. The request may already hold a batch of rows.
    fn infer(&self, request: &InferRequest) -> Result<InferResponse>;

    /// Approximate resident size, used by the cache to report memory use.
    fn size_bytes(&self) -> usize;
}

/// Creates backends of one kind from model artifacts.
///
/// Adding a runtime means adding a factory and registering it; no existing code
/// changes.
pub trait BackendFactory: Send + Sync {
    /// The platform string this factory serves, e.g. `"onnx_onnxv1"`.
    fn platform(&self) -> &'static str;

    /// Whether this factory can load the given artifact.
    fn accepts(&self, artifact: &ModelArtifact) -> bool;

    /// Loads the artifact into a running backend.
    fn load(&self, artifact: &ModelArtifact) -> Result<Arc<dyn Backend>>;
}

/// Read access to the model repository.
#[async_trait]
pub trait ModelStore: Send + Sync {
    /// Every model version the store holds.
    async fn list(&self) -> Result<Vec<ModelKey>>;

    /// Fetches a model's bytes and settings.
    async fn fetch(&self, key: &ModelKey) -> Result<ModelArtifact>;
}

/// Write access, implemented by stores that can be populated from elsewhere.
#[async_trait]
pub trait ModelWriter: Send + Sync {
    /// Stores an artifact, replacing any previous copy of the same version.
    async fn put(&self, artifact: &ModelArtifact) -> Result<()>;

    /// The digest already stored for a key, if any. Lets an ingest skip
    /// artifacts whose bytes have not changed.
    async fn digest(&self, key: &ModelKey) -> Result<Option<String>>;
}

/// How recently and how often each model was used.
///
/// Persisting this lets the cache's eviction order survive a restart.
#[async_trait]
pub trait UsageIndex: Send + Sync {
    /// Records that a model was used at `at_unix_ms`.
    async fn touch(&self, key: &ModelKey, at_unix_ms: u64, size_bytes: usize) -> Result<()>;

    /// Reads back the recorded usage, newest first.
    async fn usage(&self) -> Result<Vec<ModelUsage>>;
}

/// A persisted usage record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelUsage {
    pub key: ModelKey,
    pub last_used_unix_ms: u64,
    pub load_count: u64,
    pub size_bytes: usize,
}

/// A cache of loaded backends.
#[async_trait]
pub trait ModelCache: Send + Sync {
    /// Returns the loaded backend for `key`, loading it on a miss.
    ///
    /// Concurrent misses for the same key must load the model only once.
    async fn get(&self, key: &ModelKey) -> Result<Arc<dyn Backend>>;

    /// Drops a model from the cache, if present.
    async fn invalidate(&self, key: &ModelKey);

    /// Evicts every entry that has outlived its idle time-to-live.
    async fn sweep(&self) -> usize;

    fn stats(&self) -> CacheStats;
}

/// Counters describing cache behaviour.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub entries: usize,
    pub hits: u64,
    pub misses: u64,
    /// Entries dropped because the cache was full.
    pub evictions: u64,
    /// Entries dropped because they sat unused past the time-to-live.
    pub expirations: u64,
    pub resident_bytes: usize,
}

/// A source of monotonic time, injectable so that time-dependent behaviour such
/// as the cache time-to-live can be tested without sleeping.
pub trait Clock: Send + Sync + std::fmt::Debug {
    /// Milliseconds since an arbitrary fixed point; never decreases.
    fn now_millis(&self) -> u64;

    /// Milliseconds since the Unix epoch, for records that outlive the process.
    fn unix_millis(&self) -> u64;
}

/// A source of Prometheus metrics text.
///
/// Kept apart from [`Inference`] because serving metrics is not part of the
/// Predict Protocol: a transport can expose it or not without the protocol
/// implementation knowing either way.
#[async_trait]
pub trait MetricsSource: Send + Sync {
    /// The current readings, in Prometheus text exposition format.
    async fn prometheus_text(&self) -> String;
}

/// Executes a request against a model, hiding resolution, caching and batching.
#[async_trait]
pub trait Inference: Send + Sync {
    async fn infer(&self, request: InferRequest) -> Result<InferResponse>;

    /// Metadata for a model, resolving the version if none was given.
    async fn metadata(&self, model: &crate::model::ModelRef) -> Result<ModelMetadata>;

    /// Whether a model is loaded or loadable right now.
    async fn is_model_ready(&self, model: &crate::model::ModelRef) -> bool;

    /// Whether every known model is ready.
    async fn is_ready(&self) -> bool;

    /// Every model version the server knows about.
    async fn models(&self) -> Result<Vec<ModelKey>>;
}
