//! The inference service: resolution, caching and batching behind one trait.

use crate::registry::ModelRegistry;
use async_trait::async_trait;
use galemind_batching::{BatchPolicy, Dispatcher, DispatcherConfig, DispatcherSnapshot};
use galemind_core::{
    CacheStats, Clock, InferError, InferRequest, InferResponse, Inference, ModelCache, ModelKey,
    ModelMetadata, ModelRef,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

type Result<T> = std::result::Result<T, InferError>;

/// Serves inference for every model in the repository.
///
/// Requests are routed by resolved model version: each version gets its own
/// batching dispatcher, created on first use, so models are demultiplexed
/// rather than sharing one queue.
pub struct InferenceService {
    registry: Arc<ModelRegistry>,
    cache: Arc<dyn ModelCache>,
    policy: Arc<dyn BatchPolicy>,
    clock: Arc<dyn Clock>,
    dispatcher_config: DispatcherConfig,
    dispatchers: RwLock<HashMap<ModelKey, Arc<Dispatcher>>>,
    ready: AtomicBool,
}

impl InferenceService {
    pub fn new(
        registry: Arc<ModelRegistry>,
        cache: Arc<dyn ModelCache>,
        policy: Arc<dyn BatchPolicy>,
        clock: Arc<dyn Clock>,
        dispatcher_config: DispatcherConfig,
    ) -> Self {
        Self {
            registry,
            cache,
            policy,
            clock,
            dispatcher_config,
            dispatchers: RwLock::new(HashMap::new()),
            ready: AtomicBool::new(false),
        }
    }

    pub fn registry(&self) -> &Arc<ModelRegistry> {
        &self.registry
    }

    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// Per-model batching counters, for metrics and diagnostics.
    pub async fn dispatcher_metrics(&self) -> Vec<(ModelKey, DispatcherSnapshot)> {
        let dispatchers = self.dispatchers.read().await;
        let mut metrics: Vec<_> = dispatchers
            .iter()
            .map(|(key, dispatcher)| (key.clone(), dispatcher.metrics()))
            .collect();
        metrics.sort_by(|a, b| a.0.cmp(&b.0));
        metrics
    }

    /// Reads the repository and marks the server ready to serve.
    ///
    /// With `preload`, every model is loaded before readiness is reported, so a
    /// readiness probe passing means no request will pay a load. Without it,
    /// models load on first use.
    pub async fn start(&self, preload: bool) -> Result<()> {
        self.registry.refresh().await?;

        if preload {
            for key in self.registry.keys().await {
                match self.cache.get(&key).await {
                    Ok(_) => debug!(model = %key, "preloaded model"),
                    Err(e) => {
                        warn!(model = %key, error = %e, "could not preload model");
                        return Err(e);
                    }
                }
            }
            info!("preloaded every model in the repository");
        }

        self.ready.store(true, Ordering::SeqCst);
        Ok(())
    }

    /// Evicts models that have outlived their time-to-live.
    pub async fn sweep_cache(&self) -> usize {
        self.cache.sweep().await
    }

    /// Stops every dispatcher, letting queued requests finish first.
    pub async fn shutdown(&self) {
        self.ready.store(false, Ordering::SeqCst);
        let dispatchers: Vec<_> = self.dispatchers.write().await.drain().collect();
        for (key, dispatcher) in dispatchers {
            match Arc::try_unwrap(dispatcher) {
                Ok(dispatcher) => dispatcher.shutdown().await,
                // A request still holds this dispatcher; dropping our handle is
                // enough, its queue closes once that request completes.
                Err(_) => debug!(model = %key, "dispatcher still in use at shutdown"),
            }
        }
    }

    /// The dispatcher for a model version, started on first use.
    async fn dispatcher(&self, key: &ModelKey) -> Arc<Dispatcher> {
        if let Some(dispatcher) = self.dispatchers.read().await.get(key) {
            return Arc::clone(dispatcher);
        }

        let mut dispatchers = self.dispatchers.write().await;
        // Another task may have started it while we waited for the lock.
        Arc::clone(dispatchers.entry(key.clone()).or_insert_with(|| {
            info!(model = %key, "starting batching dispatcher");
            Arc::new(Dispatcher::spawn(
                key.clone(),
                Arc::clone(&self.cache),
                Arc::clone(&self.policy),
                Arc::clone(&self.clock),
                self.dispatcher_config,
            ))
        }))
    }
}

#[async_trait]
impl Inference for InferenceService {
    async fn infer(&self, request: InferRequest) -> Result<InferResponse> {
        let key = self.registry.resolve(&request.model).await?;

        // Loading the model also gives the metadata the request is checked
        // against, so a malformed request is refused before it is queued.
        let backend = self.cache.get(&key).await?;
        request.validate_against(backend.metadata())?;
        drop(backend);

        let dispatcher = self.dispatcher(&key).await;
        let mut response = dispatcher.submit(request, self.clock.now_millis()).await?;

        // The response names the version actually served, as the protocol asks.
        response.model_name = key.name.clone();
        response.model_version = key.version.clone();
        Ok(response)
    }

    async fn metadata(&self, model: &ModelRef) -> Result<ModelMetadata> {
        let key = self.registry.resolve(model).await?;
        let backend = self.cache.get(&key).await?;
        let mut metadata = backend.metadata().clone();

        // The backend only knows the version it was loaded from; the registry
        // knows which versions the server can serve.
        metadata.name = key.name.clone();
        if let Some(versions) = self.registry.versions(&key.name).await {
            metadata.versions = versions;
        }
        Ok(metadata)
    }

    async fn is_model_ready(&self, model: &ModelRef) -> bool {
        match self.registry.resolve(model).await {
            Ok(key) => self.cache.get(&key).await.is_ok(),
            Err(_) => false,
        }
    }

    async fn is_ready(&self) -> bool {
        self.ready.load(Ordering::SeqCst) && !self.registry.is_empty().await
    }

    async fn models(&self) -> Result<Vec<ModelKey>> {
        Ok(self.registry.keys().await)
    }
}
