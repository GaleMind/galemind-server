//! The inference service: resolution, caching and batching behind one trait.

use crate::metrics::{render_prometheus, MetricsRegistry, PrometheusReport};
use crate::registry::ModelRegistry;
use crate::resources::{ResourceSampler, ResourceSnapshot};
use async_trait::async_trait;
use galemind_batching::{BatchPolicy, Dispatcher, DispatcherConfig, DispatcherSnapshot};
use galemind_core::{
    CacheStats, Clock, InferError, InferRequest, InferResponse, Inference, MetricsSource,
    ModelCache, ModelKey, ModelMetadata, ModelRef, SERVER_VERSION,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
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
    metrics: Arc<MetricsRegistry>,
    resources: ResourceSampler,
    /// The newest resource reading. Utilisation is a rate, so it is sampled on
    /// an interval rather than computed inside a metrics scrape.
    latest_resources: RwLock<ResourceSnapshot>,
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
            metrics: Arc::new(MetricsRegistry::new()),
            resources: ResourceSampler::new(),
            latest_resources: RwLock::new(ResourceSnapshot::default()),
        }
    }

    pub fn registry(&self) -> &Arc<ModelRegistry> {
        &self.registry
    }

    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// Per-model request counters.
    pub fn metrics(&self) -> &Arc<MetricsRegistry> {
        &self.metrics
    }

    /// Takes a resource reading and keeps it as the newest.
    ///
    /// Called on the metrics interval by the server; each reading measures CPU
    /// use since the previous one.
    pub async fn sample_resources(&self) -> ResourceSnapshot {
        let snapshot = self.resources.sample();
        *self.latest_resources.write().await = snapshot;
        snapshot
    }

    /// The newest resource reading.
    pub async fn resources(&self) -> ResourceSnapshot {
        *self.latest_resources.read().await
    }

    /// Gathers everything a metrics scrape reports.
    pub async fn prometheus_report(&self) -> PrometheusReport {
        let dispatchers = self
            .dispatcher_metrics()
            .await
            .into_iter()
            .collect::<HashMap<_, _>>();

        PrometheusReport {
            server_version: SERVER_VERSION,
            models: self.metrics.snapshot().await,
            dispatchers,
            unresolved_failures: self.metrics.unresolved_failures(),
            cache: self.cache.stats(),
            resources: self.resources().await,
            loaded_models: self.registry.keys().await.len(),
        }
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

    /// Resolves, validates and runs a request, reporting the model it resolved
    /// to so that the outcome can be counted against it.
    ///
    /// A request that never resolves has no model to be counted against, which
    /// is why the model is part of both the success and the failure.
    async fn infer_inner(
        &self,
        request: InferRequest,
    ) -> std::result::Result<(ModelKey, InferResponse), (Option<ModelKey>, InferError)> {
        let key = self
            .registry
            .resolve(&request.model)
            .await
            .map_err(|e| (None, e))?;

        // Loading the model also gives the metadata the request is checked
        // against, so a malformed request is refused before it is queued.
        let backend = self
            .cache
            .get(&key)
            .await
            .map_err(|e| (Some(key.clone()), e))?;
        if let Err(e) = request.validate_against(backend.metadata()) {
            return Err((Some(key), e));
        }
        drop(backend);

        let dispatcher = self.dispatcher(&key).await;
        let mut response = dispatcher
            .submit(request, self.clock.now_millis())
            .await
            .map_err(|e| (Some(key.clone()), e))?;

        // The response names the version actually served, as the protocol asks.
        response.model_name = key.name.clone();
        response.model_version = key.version.clone();
        Ok((key, response))
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
        // Timing covers everything the server does with the request, including
        // resolving and validating it, because that is what a caller waits for.
        let started = Instant::now();
        let outcome = self.infer_inner(request).await;

        let elapsed = started.elapsed();
        match &outcome {
            Ok((key, _)) => self.metrics.record_success(key, elapsed).await,
            Err((key, error)) => {
                self.metrics
                    .record_failure(key.as_ref(), error, elapsed)
                    .await
            }
        }

        outcome.map(|(_, response)| response).map_err(|(_, e)| e)
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

#[async_trait]
impl MetricsSource for InferenceService {
    async fn prometheus_text(&self) -> String {
        render_prometheus(&self.prometheus_report().await)
    }
}
