//! A bounded, time-to-live model cache with single-flight loading.

use async_trait::async_trait;
use galemind_core::{
    Backend, BackendRegistry, CacheStats, Clock, InferError, ModelCache, ModelKey, ModelStore,
    UsageIndex,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tracing::{debug, info};

type Result<T> = std::result::Result<T, InferError>;

/// How the cache is bounded.
#[derive(Debug, Clone, Copy)]
pub struct LruTtlConfig {
    /// Most models kept loaded at once. The least recently used is dropped when
    /// a new model needs room.
    pub capacity: usize,
    /// How long a model may sit unused before it is unloaded.
    pub ttl: std::time::Duration,
}

impl Default for LruTtlConfig {
    fn default() -> Self {
        Self {
            capacity: 8,
            ttl: std::time::Duration::from_secs(3_600),
        }
    }
}

struct Entry {
    backend: Arc<dyn Backend>,
    last_used_millis: u64,
}

#[derive(Default)]
struct Counters {
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    expirations: AtomicU64,
}

/// Caches loaded backends, loading them from a store on a miss.
///
/// Concurrent misses for the same model wait on one load rather than each
/// building their own session: a per-key gate is taken before loading and the
/// cache is re-checked after it is acquired.
pub struct LruTtlCache {
    store: Arc<dyn ModelStore>,
    registry: Arc<BackendRegistry>,
    usage: Option<Arc<dyn UsageIndex>>,
    clock: Arc<dyn Clock>,
    config: LruTtlConfig,
    entries: Mutex<HashMap<ModelKey, Entry>>,
    gates: Mutex<HashMap<ModelKey, Arc<tokio::sync::Mutex<()>>>>,
    counters: Counters,
}

impl LruTtlCache {
    pub fn new(
        store: Arc<dyn ModelStore>,
        registry: Arc<BackendRegistry>,
        clock: Arc<dyn Clock>,
        config: LruTtlConfig,
    ) -> Self {
        Self {
            store,
            registry,
            usage: None,
            clock,
            config,
            entries: Mutex::new(HashMap::new()),
            gates: Mutex::new(HashMap::new()),
            counters: Counters::default(),
        }
    }

    /// Records every load in a usage index, so eviction order survives restarts.
    pub fn with_usage_index(mut self, usage: Arc<dyn UsageIndex>) -> Self {
        self.usage = Some(usage);
        self
    }

    fn ttl_millis(&self) -> u64 {
        self.config.ttl.as_millis() as u64
    }

    fn lock_entries(&self) -> Result<std::sync::MutexGuard<'_, HashMap<ModelKey, Entry>>> {
        self.entries
            .lock()
            .map_err(|_| InferError::Internal("model cache lock poisoned".to_string()))
    }

    /// Returns a cached backend, refreshing its last-used stamp.
    ///
    /// An entry already past its time-to-live is dropped and reported as a miss,
    /// so a model is never served from an entry the sweeper would have removed.
    fn take_live(&self, key: &ModelKey, now: u64) -> Result<Option<Arc<dyn Backend>>> {
        let mut entries = self.lock_entries()?;
        let Some(entry) = entries.get_mut(key) else {
            return Ok(None);
        };

        if now.saturating_sub(entry.last_used_millis) >= self.ttl_millis() {
            entries.remove(key);
            self.counters.expirations.fetch_add(1, Ordering::Relaxed);
            debug!(model = %key, "model expired on access");
            return Ok(None);
        }

        entry.last_used_millis = now;
        Ok(Some(Arc::clone(&entry.backend)))
    }

    /// Makes room for one more entry by dropping the least recently used.
    fn evict_if_full(&self, entries: &mut HashMap<ModelKey, Entry>) {
        while entries.len() >= self.config.capacity {
            let Some(victim) = entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used_millis)
                .map(|(key, _)| key.clone())
            else {
                return;
            };
            entries.remove(&victim);
            self.counters.evictions.fetch_add(1, Ordering::Relaxed);
            info!(model = %victim, "evicted least recently used model");
        }
    }

    fn gate(&self, key: &ModelKey) -> Result<Arc<tokio::sync::Mutex<()>>> {
        let mut gates = self
            .gates
            .lock()
            .map_err(|_| InferError::Internal("model cache gate lock poisoned".to_string()))?;
        Ok(Arc::clone(
            gates
                .entry(key.clone())
                .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(()))),
        ))
    }

    /// Fetches the artifact and builds a backend. The build is CPU bound, so it
    /// runs on a blocking thread.
    async fn load(&self, key: &ModelKey) -> Result<Arc<dyn Backend>> {
        let artifact = self.store.fetch(key).await?;
        let registry = Arc::clone(&self.registry);
        let model = key.clone();

        let backend = tokio::task::spawn_blocking(move || registry.load(&artifact))
            .await
            .map_err(|e| InferError::Internal(format!("loading {model} failed: {e}")))??;

        if let Some(usage) = &self.usage {
            let recorded = usage
                .touch(key, self.clock.unix_millis(), backend.size_bytes())
                .await;
            if let Err(e) = recorded {
                // A usage record is an optimisation, not a correctness concern.
                debug!(model = %key, error = %e, "could not record model usage");
            }
        }

        Ok(backend)
    }
}

#[async_trait]
impl ModelCache for LruTtlCache {
    async fn get(&self, key: &ModelKey) -> Result<Arc<dyn Backend>> {
        if let Some(backend) = self.take_live(key, self.clock.now_millis())? {
            self.counters.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(backend);
        }

        let gate = self.gate(key)?;
        let _guard = gate.lock().await;

        // Another task may have loaded this model while we waited for the gate.
        if let Some(backend) = self.take_live(key, self.clock.now_millis())? {
            self.counters.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(backend);
        }

        self.counters.misses.fetch_add(1, Ordering::Relaxed);
        let backend = self.load(key).await?;

        let mut entries = self.lock_entries()?;
        self.evict_if_full(&mut entries);
        entries.insert(
            key.clone(),
            Entry {
                backend: Arc::clone(&backend),
                last_used_millis: self.clock.now_millis(),
            },
        );
        info!(model = %key, entries = entries.len(), "loaded model into cache");

        Ok(backend)
    }

    async fn invalidate(&self, key: &ModelKey) {
        if let Ok(mut entries) = self.lock_entries() {
            if entries.remove(key).is_some() {
                debug!(model = %key, "invalidated cached model");
            }
        }
    }

    async fn sweep(&self) -> usize {
        let now = self.clock.now_millis();
        let ttl = self.ttl_millis();

        let Ok(mut entries) = self.lock_entries() else {
            return 0;
        };
        let expired: Vec<_> = entries
            .iter()
            .filter(|(_, entry)| now.saturating_sub(entry.last_used_millis) >= ttl)
            .map(|(key, _)| key.clone())
            .collect();

        for key in &expired {
            entries.remove(key);
            self.counters.expirations.fetch_add(1, Ordering::Relaxed);
            info!(model = %key, "unloaded model after time-to-live expired");
        }
        expired.len()
    }

    fn stats(&self) -> CacheStats {
        let (entries, resident_bytes) = self
            .lock_entries()
            .map(|entries| {
                (
                    entries.len(),
                    entries.values().map(|e| e.backend.size_bytes()).sum(),
                )
            })
            .unwrap_or((0, 0));

        CacheStats {
            entries,
            hits: self.counters.hits.load(Ordering::Relaxed),
            misses: self.counters.misses.load(Ordering::Relaxed),
            evictions: self.counters.evictions.load(Ordering::Relaxed),
            expirations: self.counters.expirations.load(Ordering::Relaxed),
            resident_bytes,
        }
    }
}
