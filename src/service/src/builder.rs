//! Assembling an inference service from its parts.

use crate::registry::ModelRegistry;
use crate::runtime::RuntimeInfo;
use crate::service::InferenceService;
use galemind_batching::{AdaptiveBatchPolicy, BatchPolicy, DispatcherConfig};
use galemind_cache::{LruTtlCache, LruTtlConfig};
use galemind_core::{
    BackendRegistry, Clock, InferError, ModelCache, ModelStore, ModelWriter, SystemClock,
    UsageIndex,
};
use galemind_store::{ingest, FsModelStore, IngestReport};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

type Result<T> = std::result::Result<T, InferError>;

/// Wires a service together, choosing defaults for anything not given.
///
/// Every part is injected as a trait object, so a test can supply a memory
/// store and a controllable clock while production supplies RocksDB and the
/// real one, with no change to the service itself.
pub struct ServiceBuilder {
    backends: BackendRegistry,
    store: Option<Arc<dyn ModelStore>>,
    usage: Option<Arc<dyn UsageIndex>>,
    policy: Option<Arc<dyn BatchPolicy>>,
    clock: Arc<dyn Clock>,
    cache_config: LruTtlConfig,
    dispatcher_config: DispatcherConfig,
    runtime: Option<RuntimeInfo>,
}

impl ServiceBuilder {
    pub fn new(backends: BackendRegistry) -> Self {
        Self {
            backends,
            store: None,
            usage: None,
            policy: None,
            clock: Arc::new(SystemClock::new()),
            cache_config: LruTtlConfig::default(),
            dispatcher_config: DispatcherConfig::default(),
            runtime: None,
        }
    }

    /// Serves models straight from a directory tree.
    pub fn with_filesystem_store(mut self, root: impl Into<PathBuf>) -> Self {
        self.store = Some(Arc::new(FsModelStore::new(root)));
        self
    }

    pub fn with_store(mut self, store: Arc<dyn ModelStore>) -> Self {
        self.store = Some(store);
        self
    }

    pub fn with_usage_index(mut self, usage: Arc<dyn UsageIndex>) -> Self {
        self.usage = Some(usage);
        self
    }

    pub fn with_policy(mut self, policy: Arc<dyn BatchPolicy>) -> Self {
        self.policy = Some(policy);
        self
    }

    pub fn with_clock(mut self, clock: Arc<dyn Clock>) -> Self {
        self.clock = clock;
        self
    }

    pub fn with_cache_config(mut self, config: LruTtlConfig) -> Self {
        self.cache_config = config;
        self
    }

    pub fn with_dispatcher_config(mut self, config: DispatcherConfig) -> Self {
        self.dispatcher_config = config;
        self
    }

    /// Records the thread plan the binary actually started with, so that it is
    /// reported in the metrics rather than guessed at from the machine.
    pub fn with_runtime_info(mut self, runtime: RuntimeInfo) -> Self {
        self.runtime = Some(runtime);
        self
    }

    pub fn build(self) -> Result<Arc<InferenceService>> {
        if self.backends.is_empty() {
            return Err(InferError::Internal(
                "no backend factories were registered".to_string(),
            ));
        }
        let store = self
            .store
            .ok_or_else(|| InferError::Internal("no model store was configured".to_string()))?;

        let mut cache = LruTtlCache::new(
            Arc::clone(&store),
            Arc::new(self.backends),
            Arc::clone(&self.clock),
            self.cache_config,
        );
        if let Some(usage) = self.usage {
            cache = cache.with_usage_index(usage);
        }

        let policy = self
            .policy
            .unwrap_or_else(|| Arc::new(AdaptiveBatchPolicy::default()));

        info!(
            cache_capacity = self.cache_config.capacity,
            cache_ttl_secs = self.cache_config.ttl.as_secs(),
            queue_capacity = self.dispatcher_config.queue_capacity,
            "built inference service"
        );

        let mut service = InferenceService::new(
            Arc::new(ModelRegistry::new(Arc::clone(&store))),
            store,
            Arc::new(cache) as Arc<dyn ModelCache>,
            policy,
            self.clock,
            self.dispatcher_config,
        );
        if let Some(runtime) = self.runtime {
            service.set_runtime_info(runtime);
        }
        Ok(Arc::new(service))
    }
}

/// Copies a filesystem model repository into a durable store.
///
/// This is the startup path when RocksDB is the serving store: the directory
/// tree is the source of truth operators edit, RocksDB is what the server
/// reads, and unchanged artifacts are not rewritten.
pub async fn ingest_repository<T>(root: impl Into<PathBuf>, target: &T) -> Result<IngestReport>
where
    T: ModelStore + ModelWriter + ?Sized,
{
    let source = FsModelStore::new(root);
    ingest(&source, target).await
}
