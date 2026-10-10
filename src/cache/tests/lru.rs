//! Behaviour of the hot-tier model cache.

use galemind_cache::{LruTtlCache, LruTtlConfig};
use galemind_core::{
    Backend, BackendFactory, BackendRegistry, InferError, InferRequest, InferResponse,
    ModelArtifact, ModelCache, ModelConfig, ModelKey, ModelMetadata, ModelStore, ModelWriter,
    TestClock, UsageIndex, PLATFORM_ONNX,
};
use galemind_store::MemoryModelStore;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

struct StubBackend {
    metadata: ModelMetadata,
}

impl Backend for StubBackend {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }
    fn infer(&self, _request: &InferRequest) -> Result<InferResponse, InferError> {
        unimplemented!("the cache never runs inference")
    }
    fn size_bytes(&self) -> usize {
        1_024
    }
}

/// Counts how often a model is actually built, which is how single-flight and
/// eviction are observed from the outside.
struct CountingFactory {
    loads: Arc<AtomicUsize>,
    load_delay: Duration,
}

impl BackendFactory for CountingFactory {
    fn platform(&self) -> &'static str {
        PLATFORM_ONNX
    }

    fn accepts(&self, artifact: &ModelArtifact) -> bool {
        artifact.platform == PLATFORM_ONNX
    }

    fn load(&self, artifact: &ModelArtifact) -> Result<Arc<dyn Backend>, InferError> {
        if !self.load_delay.is_zero() {
            std::thread::sleep(self.load_delay);
        }
        self.loads.fetch_add(1, Ordering::SeqCst);
        Ok(Arc::new(StubBackend {
            metadata: ModelMetadata {
                name: artifact.key.name.clone(),
                versions: vec![artifact.key.version.clone()],
                platform: PLATFORM_ONNX.to_string(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                max_batch_size: Some(8),
            },
        }))
    }
}

fn artifact(name: &str) -> ModelArtifact {
    ModelArtifact {
        key: ModelKey::new(name, "1"),
        platform: PLATFORM_ONNX.to_string(),
        digest: format!("digest-{name}"),
        config: ModelConfig::default(),
        bytes: vec![0u8; 16],
    }
}

struct Fixture {
    cache: LruTtlCache,
    loads: Arc<AtomicUsize>,
    clock: Arc<TestClock>,
    store: Arc<MemoryModelStore>,
}

impl Fixture {
    /// How many times a backend was actually built.
    fn loads(&self) -> usize {
        self.loads.load(Ordering::SeqCst)
    }
}

async fn fixture(models: &[&str], config: LruTtlConfig, load_delay: Duration) -> Fixture {
    let store = Arc::new(MemoryModelStore::new());
    for name in models {
        store.put(&artifact(name)).await.unwrap();
    }

    let loads = Arc::new(AtomicUsize::new(0));
    let registry = Arc::new(BackendRegistry::new().with(Arc::new(CountingFactory {
        loads: Arc::clone(&loads),
        load_delay,
    })));
    let clock = Arc::new(TestClock::new(0));

    Fixture {
        cache: LruTtlCache::new(
            Arc::clone(&store) as Arc<dyn ModelStore>,
            registry,
            Arc::clone(&clock) as Arc<dyn galemind_core::Clock>,
            config,
        ),
        loads,
        clock,
        store,
    }
}

fn config(capacity: usize, ttl_secs: u64) -> LruTtlConfig {
    LruTtlConfig {
        capacity,
        ttl: Duration::from_secs(ttl_secs),
    }
}

#[tokio::test]
async fn the_default_time_to_live_is_one_hour() {
    assert_eq!(LruTtlConfig::default().ttl, Duration::from_secs(3_600));
}

#[tokio::test]
async fn a_model_is_loaded_once_and_then_served_from_memory() {
    let f = fixture(&["energy"], config(4, 3_600), Duration::ZERO).await;
    let key = ModelKey::new("energy", "1");

    f.cache.get(&key).await.unwrap();
    f.cache.get(&key).await.unwrap();
    f.cache.get(&key).await.unwrap();

    assert_eq!(f.loads(), 1);
    let stats = f.cache.stats();
    assert_eq!((stats.hits, stats.misses, stats.entries), (2, 1, 1));
    assert_eq!(stats.resident_bytes, 1_024);
}

#[tokio::test]
async fn a_model_unused_for_longer_than_the_time_to_live_is_unloaded() {
    let f = fixture(&["energy"], config(4, 3_600), Duration::ZERO).await;
    let key = ModelKey::new("energy", "1");
    f.cache.get(&key).await.unwrap();

    f.clock.advance_secs(3_599);
    assert_eq!(f.cache.sweep().await, 0, "not yet expired");

    f.clock.advance_secs(1);
    assert_eq!(f.cache.sweep().await, 1);
    assert_eq!(f.cache.stats().entries, 0);
    assert_eq!(f.cache.stats().expirations, 1);

    f.cache.get(&key).await.unwrap();
    assert_eq!(f.loads(), 2, "an expired model is loaded again");
}

#[tokio::test]
async fn using_a_model_keeps_it_alive() {
    let f = fixture(&["energy"], config(4, 3_600), Duration::ZERO).await;
    let key = ModelKey::new("energy", "1");

    for _ in 0..4 {
        f.clock.advance_secs(3_000);
        f.cache.get(&key).await.unwrap();
        assert_eq!(f.cache.sweep().await, 0);
    }
    assert_eq!(f.loads(), 1);
}

#[tokio::test]
async fn an_expired_entry_is_never_served_even_before_a_sweep() {
    let f = fixture(&["energy"], config(4, 60), Duration::ZERO).await;
    let key = ModelKey::new("energy", "1");
    f.cache.get(&key).await.unwrap();

    f.clock.advance_secs(61);
    f.cache.get(&key).await.unwrap();

    assert_eq!(f.loads(), 2);
    assert_eq!(f.cache.stats().expirations, 1);
}

#[tokio::test]
async fn filling_the_cache_evicts_the_least_recently_used_model() {
    let f = fixture(&["a", "b", "c"], config(2, 3_600), Duration::ZERO).await;
    let (a, b, c) = (
        ModelKey::new("a", "1"),
        ModelKey::new("b", "1"),
        ModelKey::new("c", "1"),
    );

    f.cache.get(&a).await.unwrap();
    f.clock.advance_secs(1);
    f.cache.get(&b).await.unwrap();
    f.clock.advance_secs(1);
    // Touching "a" makes "b" the least recently used.
    f.cache.get(&a).await.unwrap();
    f.clock.advance_secs(1);
    f.cache.get(&c).await.unwrap();

    assert_eq!(f.cache.stats().entries, 2);
    assert_eq!(f.cache.stats().evictions, 1);

    f.cache.get(&a).await.unwrap();
    assert_eq!(f.loads(), 3, "'a' was kept, so nothing reloaded");

    f.cache.get(&b).await.unwrap();
    assert_eq!(f.loads(), 4, "'b' was the evicted one");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_misses_for_one_model_load_it_only_once() {
    let f = Arc::new(fixture(&["energy"], config(4, 3_600), Duration::from_millis(50)).await);
    let key = ModelKey::new("energy", "1");

    let waiters: Vec<_> = (0..16)
        .map(|_| {
            let f = Arc::clone(&f);
            let key = key.clone();
            tokio::spawn(async move { f.cache.get(&key).await.map(|_| ()) })
        })
        .collect();

    for waiter in waiters {
        waiter.await.unwrap().unwrap();
    }

    assert_eq!(f.loads(), 1, "a single-flight load was expected");
    assert_eq!(f.cache.stats().misses, 1);
}

#[tokio::test]
async fn invalidating_a_model_forces_the_next_request_to_reload_it() {
    let f = fixture(&["energy"], config(4, 3_600), Duration::ZERO).await;
    let key = ModelKey::new("energy", "1");

    f.cache.get(&key).await.unwrap();
    f.cache.invalidate(&key).await;
    assert_eq!(f.cache.stats().entries, 0);

    f.cache.get(&key).await.unwrap();
    assert_eq!(f.loads(), 2);
}

#[tokio::test]
async fn a_model_the_store_does_not_hold_is_reported_not_found() {
    let f = fixture(&[], config(4, 3_600), Duration::ZERO).await;
    let Err(err) = f.cache.get(&ModelKey::new("ghost", "1")).await else {
        panic!("an unknown model must not load");
    };
    assert!(matches!(err, InferError::VersionNotFound { .. }), "{err}");
    assert_eq!(f.cache.stats().entries, 0);
}

#[tokio::test]
async fn every_load_is_recorded_in_the_usage_index() {
    let f = fixture(&["energy"], config(1, 3_600), Duration::ZERO).await;
    let usage_store = Arc::clone(&f.store);
    let cache = f
        .cache
        .with_usage_index(Arc::clone(&usage_store) as Arc<dyn UsageIndex>);
    let key = ModelKey::new("energy", "1");

    f.clock.advance_secs(5);
    cache.get(&key).await.unwrap();
    cache.invalidate(&key).await;
    f.clock.advance_secs(5);
    cache.get(&key).await.unwrap();

    let usage = usage_store.usage().await.unwrap();
    assert_eq!(usage.len(), 1);
    assert_eq!(usage[0].key, key);
    assert_eq!(usage[0].load_count, 2);
    assert_eq!(usage[0].size_bytes, 1_024);
    assert_eq!(usage[0].last_used_unix_ms, 10_000);
}
