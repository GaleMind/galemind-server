//! The GaleMind server binary.

#![forbid(unsafe_code)]

mod cli;

/// jemalloc in place of glibc's allocator, when built with the `jemalloc`
/// feature.
///
/// The reason is reclaim, not speed. glibc keeps freed memory in per-thread
/// arenas and has no way to hand it back short of `malloc_trim`, which is an
/// FFI call this crate cannot make; measured here, a server with every model
/// unloaded and no work in flight still held 70.8 MB against 16.2 MB at
/// startup, all of it retained by the allocator. jemalloc returns unused
/// pages on a decay timer instead, which is what makes an idle server give
/// its memory back -- and it does the unsafe inside its own crate, so this
/// one stays `forbid(unsafe_code)`.
///
/// The decay is configured through jemalloc's own options rather than ours,
/// because it reads them before `main` runs: `dirty_decay_ms` and
/// `muzzy_decay_ms` say how long a page may sit unused before being returned,
/// and `background_thread` is what returns it without waiting for the next
/// allocation to notice.
#[cfg(feature = "jemalloc")]
#[global_allocator]
static ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// What the metrics call the allocator this binary installed.
#[cfg(feature = "jemalloc")]
const ALLOCATOR_NAME: &str = "jemalloc";

/// No `#[global_allocator]`, so whatever the platform provides.
#[cfg(not(feature = "jemalloc"))]
const ALLOCATOR_NAME: &str = galemind_service::SYSTEM_ALLOCATOR;

/// Returns jemalloc's retained pages on demand, for the memory guard.
///
/// jemalloc hands pages back once they have gone unused for
/// `dirty_decay_ms`, which is right for a server that is merely idle and too
/// slow for one that is over a memory limit now. Writing a decay of zero
/// purges what is eligible immediately; the configured interval is then put
/// back, so a reclaim borrows the aggressive setting rather than switching to
/// it.
///
/// `arena.4096.*` is jemalloc's "every arena" index. The writes go through
/// the typed `mallctl` API, so none of this needs `unsafe`: the FFI is inside
/// jemalloc's own crate, as it is for the allocator itself.
#[cfg(feature = "jemalloc")]
struct JemallocReclaim;

#[cfg(feature = "jemalloc")]
impl JemallocReclaim {
    /// jemalloc's index meaning "every arena".
    const ALL_ARENAS: u32 = 4096;

    /// Asks every arena to purge one kind of page, and restores the window.
    ///
    /// `dirty` pages are freed but still mapped; `muzzy` ones have been
    /// madvised away without being unmapped. Both have to be asked for,
    /// since returning one and keeping the other halves the effect.
    ///
    /// The all-arenas index is tried first and the real arenas one by one if
    /// it refuses, which it does under load. A partial purge is still a
    /// purge, so this reports how many arenas accepted it rather than
    /// failing on the first that did not.
    fn purge(key: &str, configured: &[u8]) -> Result<usize, String> {
        use tikv_jemalloc_ctl::{Access, AsName};

        // What to go back to, read from jemalloc's own options rather than
        // from a copy of them kept here.
        let window: isize = configured.name().read().unwrap_or(10_000);
        let set = |arena: u32, ms: isize| -> Result<(), String> {
            let name = format!("arena.{arena}.{key}\0");
            name.as_bytes()
                .name()
                .write(ms)
                .map_err(|e| format!("arena.{arena}.{key}: {e}"))
        };

        let mut purged = 0;
        let mut last = String::new();
        match set(Self::ALL_ARENAS, 0) {
            Ok(()) => purged += 1,
            Err(e) => {
                last = e;
                // Only arenas that exist accept this, and which exist
                // depends on how many threads have allocated, so a refusal
                // per unused arena is expected rather than a failure.
                let narenas: u32 = b"arenas.narenas\0".name().read().unwrap_or(0);
                for arena in 0..narenas {
                    if set(arena, 0).is_ok() {
                        purged += 1;
                    }
                }
            }
        }

        // Put the window back, whatever happened: leaving a decay of zero in
        // place would purge after every free, which is the opposite of what
        // the allocator was chosen for.
        let window = window.max(0);
        let _ = set(Self::ALL_ARENAS, window);
        let narenas: u32 = b"arenas.narenas\0".name().read().unwrap_or(0);
        for arena in 0..narenas {
            let _ = set(arena, window);
        }

        if purged > 0 {
            Ok(purged)
        } else {
            Err(last)
        }
    }
}

#[cfg(feature = "jemalloc")]
impl galemind_service::AllocatorReclaim for JemallocReclaim {
    fn name(&self) -> &'static str {
        "jemalloc"
    }

    fn reclaim(&self) -> galemind_service::Reclaim {
        let dirty = Self::purge("dirty_decay_ms", b"opt.dirty_decay_ms\0");
        let muzzy = Self::purge("muzzy_decay_ms", b"opt.muzzy_decay_ms\0");

        match (&dirty, &muzzy) {
            // Either kind purged is progress; both refusing is a refusal.
            (Ok(_), _) | (_, Ok(_)) => {
                debug!(?dirty, ?muzzy, "asked jemalloc to return its pages");
                galemind_service::Reclaim::Done
            }
            (Err(dirty), Err(muzzy)) => {
                galemind_service::Reclaim::Refused(format!("dirty: {dirty}; muzzy: {muzzy}"))
            }
        }
    }
}

/// Turns on the thread that returns unused pages without being asked.
///
/// jemalloc's decay is on by default -- a page unused for ten seconds is
/// eligible to go back to the operating system -- but something has to
/// notice, and without a background thread that only happens when the next
/// allocation walks past it. An idle server makes no allocations, which is
/// exactly when its memory should be going back, so the thread is enabled
/// here.
///
/// The decay interval itself is read before `main` runs, so it is set in the
/// environment rather than here:
///
/// ```text
/// _RJEM_MALLOC_CONF=dirty_decay_ms:60000,muzzy_decay_ms:60000
/// ```
#[cfg(feature = "jemalloc")]
fn start_background_reclaim() {
    match tikv_jemalloc_ctl::background_thread::write(true) {
        Ok(()) => info!(
            allocator = "jemalloc",
            "returning unused pages in the background"
        ),
        // Not fatal: without it, pages are still returned, just only when an
        // allocation happens to notice.
        Err(e) => warn!(
            allocator = "jemalloc",
            "cannot start the background reclaim thread: {e}"
        ),
    }
}

use anyhow::{Context, Result};
use clap::Parser;
use cli::{Cli, Command, RepositoryArgs, ServeArgs, StoreKind};
use galemind_batching::{AdaptiveBatchConfig, AdaptiveBatchPolicy, DispatcherConfig};
use galemind_cache::LruTtlConfig;
use galemind_core::{BackendRegistry, Inference, MetricsSource, ModelStore, SystemClock};
use galemind_onnx::{OnnxBackendFactory, OnnxRuntimeConfig};
use galemind_service::{InferenceService, RuntimeInfo, ServiceBuilder};
use galemind_store::FsModelStore;
use rest_server::{Authenticator, NoAuth, StaticTokenAuth};
use std::sync::Arc;
use tokio::signal;
use tokio::sync::broadcast;
use tracing::{debug, info, warn};
use tracing_subscriber::EnvFilter;

/// Builds the async runtime by hand rather than with `#[tokio::main]`.
///
/// The macro's defaults are one worker per core and 512 blocking threads. The
/// first is right; the second is not, for a server whose blocking pool is
/// where model code runs: a burst across ten models would start as many
/// inferences as there are queued batches, each with its own ONNX Runtime
/// threads, and spend the machine on context switching. Both are therefore
/// configurable, and the plan is logged and published in the metrics.
fn main() -> Result<()> {
    let cli = Cli::parse();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new(&cli.log))
        .with_target(true)
        .init();

    #[cfg(feature = "jemalloc")]
    start_background_reclaim();

    // Only serving is worth a tuned runtime; the one-shot commands are I/O
    // bound and take the defaults.
    let threads = match &cli.command {
        Command::Serve(args) => Some(args.threads),
        _ => None,
    };

    let mut builder = tokio::runtime::Builder::new_multi_thread();
    builder.enable_all();
    if let Some(threads) = threads {
        builder
            .worker_threads(threads.resolved_worker_threads())
            .max_blocking_threads(threads.resolved_blocking_threads())
            .thread_name("galemind-worker");
    }
    let runtime = builder.build().context("cannot start the async runtime")?;

    runtime.block_on(async move {
        match cli.command {
            Command::Serve(args) => serve(args).await,
            Command::Models(args) => list_models(args).await,
            Command::Ingest(args) => ingest(args).await,
        }
    })
}

/// Builds the store a running server reads models from.
///
/// With RocksDB the directory tree is ingested first: the tree stays the thing
/// operators edit, while the server reads artifacts, metadata and usage records
/// from one durable place.
async fn build_store(
    args: &RepositoryArgs,
) -> Result<(
    Arc<dyn ModelStore>,
    Option<Arc<dyn galemind_core::UsageIndex>>,
)> {
    match args.store {
        StoreKind::Filesystem => Ok((
            Arc::new(FsModelStore::new(&args.model_repository)) as Arc<dyn ModelStore>,
            None,
        )),
        StoreKind::Rocksdb => {
            #[cfg(feature = "rocksdb-store")]
            {
                use galemind_store::RocksModelStore;

                if let Some(parent) = args.store_path.parent() {
                    tokio::fs::create_dir_all(parent)
                        .await
                        .with_context(|| format!("cannot create {}", parent.display()))?;
                }
                let store = Arc::new(
                    RocksModelStore::open(&args.store_path)
                        .with_context(|| format!("cannot open {}", args.store_path.display()))?,
                );

                let report =
                    galemind_service::ingest_repository(&args.model_repository, store.as_ref())
                        .await
                        .context("cannot ingest the model repository")?;
                info!(
                    added = report.added,
                    updated = report.updated,
                    unchanged = report.unchanged,
                    failed = report.failed,
                    store = %args.store_path.display(),
                    "model repository ingested"
                );

                Ok((
                    Arc::clone(&store) as Arc<dyn ModelStore>,
                    Some(store as Arc<dyn galemind_core::UsageIndex>),
                ))
            }
            #[cfg(not(feature = "rocksdb-store"))]
            {
                anyhow::bail!(
                    "this build has no RocksDB support; pass --store filesystem \
                     or build with the rocksdb-store feature"
                )
            }
        }
    }
}

/// Builds the service, and the memory guard that watches it.
///
/// They are built together because the service publishes the guard's
/// counters, so the guard has to exist before the service is built.
type ServiceAndGuard = (
    Arc<InferenceService>,
    Option<Arc<galemind_service::MemoryGuard>>,
);

async fn build_service(args: &ServeArgs) -> Result<ServiceAndGuard> {
    let (store, usage) = build_store(&args.repository).await?;

    // How many model versions can be in memory at once, which is what
    // per-session operator threads multiply against. Asked of the store
    // rather than assumed: a cache that holds sixteen in front of a
    // repository of two can only have two loaded, and giving each of them a
    // sixteenth of the machine would waste most of it.
    let versions = store.list().await.map(|keys| keys.len()).unwrap_or(0);
    let loadable = match versions {
        // Nothing to serve yet; the plan will be re-derived by whoever adds
        // models and restarts, and one thread is the safe floor meanwhile.
        0 => 1,
        versions => versions.min(args.cache.cache_capacity.max(1)),
    };

    let onnx = OnnxRuntimeConfig {
        intra_op_threads: args.threads.resolved_intra_op_threads(loadable),
        inter_op_threads: args.threads.inter_op_threads,
        parallel_execution: args.threads.parallel_execution,
        cpu_mem_arena: args.threads.cpu_mem_arena,
        // One allocation plan cannot fit batches that vary in size, which is
        // every batch this server forms.
        memory_pattern: false,
    };
    let backends = BackendRegistry::new().with(Arc::new(OnnxBackendFactory::with_config(onnx)));

    let runtime = RuntimeInfo {
        worker_threads: args.threads.resolved_worker_threads(),
        blocking_threads: args.threads.resolved_blocking_threads(),
        intra_op_threads: onnx.intra_op_threads,
        loadable_models: loadable,
        cpu_mem_arena: onnx.cpu_mem_arena,
        // This crate installs the global allocator, so this crate is the only
        // one that can say which one is running.
        allocator: ALLOCATOR_NAME,
        ..RuntimeInfo::detected()
    };
    info!(
        cpu_count = runtime.cpu_count,
        worker_threads = runtime.worker_threads,
        blocking_threads = runtime.blocking_threads,
        intra_op_threads = runtime.intra_op_threads,
        // The input to the operator-thread default, so a plan that looks
        // surprising can be explained without reading the code.
        loadable_models = loadable,
        parallel_execution = onnx.parallel_execution,
        compute_threads_max = runtime.compute_threads(),
        cpu_mem_arena = runtime.cpu_mem_arena,
        platform = runtime.platform,
        "thread plan"
    );
    match runtime.malloc_arena_max {
        Some(arenas) => info!(malloc_arena_max = arenas, "allocator arenas bounded"),
        // Worth a line at the default filter: it is the largest single
        // influence on resident memory, it can only be set from outside the
        // process, and a server that was not told looks like one with a leak.
        None => info!(
            "MALLOC_ARENA_MAX is unset; glibc may keep up to 8 freed-memory \
             arenas per core, which can double resident memory under load. \
             Set MALLOC_ARENA_MAX=2 to bound it"
        ),
    }
    if runtime.compute_threads() > runtime.cpu_count * 2 {
        warn!(
            compute_threads_max = runtime.compute_threads(),
            cpu_count = runtime.cpu_count,
            "more threads may run model code than the machine has cores; \
             lower --worker-threads or --intra-op-threads if throughput \
             stops climbing"
        );
    }

    let policy = AdaptiveBatchPolicy::new(AdaptiveBatchConfig {
        min_window: args.batching.min_window(),
        max_window: args.batching.max_window(),
        max_batch_size: args.batching.max_batch_size,
        sample_window: 64,
    });

    let mut builder = ServiceBuilder::new(backends)
        .with_store(store)
        .with_clock(Arc::new(SystemClock::new()))
        .with_policy(Arc::new(policy))
        .with_cache_config(LruTtlConfig {
            capacity: args.cache.cache_capacity,
            ttl: args.cache.ttl(),
        })
        .with_dispatcher_config(DispatcherConfig {
            queue_capacity: args.batching.queue_capacity,
            queue_timeout: args.batching.queue_timeout(),
            sample_window: 64,
            max_concurrent_batches: args.batching.max_concurrent_batches,
        })
        .with_runtime_info(runtime);
    if let Some(usage) = usage {
        builder = builder.with_usage_index(usage);
    }

    // The memory guard. Its reclaim seam is implemented by whoever installed
    // the allocator, which is this crate; a build without jemalloc gets one
    // that says it cannot, so the mark is still honoured as far as unloading
    // models can honour it, and the log says why that is all it did.
    #[cfg(feature = "jemalloc")]
    let reclaim: Arc<dyn galemind_service::AllocatorReclaim> = Arc::new(JemallocReclaim);
    #[cfg(not(feature = "jemalloc"))]
    let reclaim: Arc<dyn galemind_service::AllocatorReclaim> =
        Arc::new(galemind_service::NoReclaim);

    let guard = match args.metrics.memory_high_water_bytes() {
        Some(high_water_bytes) => {
            info!(
                mark_mb = high_water_bytes / (1024 * 1024),
                cooldown_secs = args.metrics.memory_reclaim_cooldown_secs,
                allocator = reclaim.name(),
                "reclaiming memory above the high-water mark"
            );
            Some(Arc::new(galemind_service::MemoryGuard::new(
                galemind_service::MemoryGuardConfig {
                    high_water_bytes: Some(high_water_bytes),
                    cooldown: args.metrics.reclaim_cooldown(),
                    may_unload: true,
                },
                Arc::clone(&reclaim),
            )))
        }
        None => {
            info!("no memory high-water mark; memory is bounded by the cache and the allocator");
            None
        }
    };

    if let Some(guard) = &guard {
        builder = builder.with_memory_guard(Arc::clone(guard));
    }

    let service = builder.build().map_err(anyhow::Error::from)?;
    service
        .start(args.preload)
        .await
        .context("cannot start the inference service")?;

    let models = service.models().await.map_err(anyhow::Error::from)?;
    if models.is_empty() {
        warn!(
            repository = %args.repository.model_repository.display(),
            "no models found; the server will report itself not ready"
        );
    } else {
        for key in &models {
            info!(model = %key, "serving model");
        }
    }

    Ok((service, guard))
}

async fn serve(args: ServeArgs) -> Result<()> {
    let (service, guard) = build_service(&args).await?;

    // One shutdown signal fans out to every task, so Ctrl-C drains the queues
    // instead of cutting requests off mid-flight.
    let (shutdown, _) = broadcast::channel::<()>(1);
    let sweeper = tokio::spawn(sweep_cache(
        Arc::clone(&service),
        args.cache.sweep_interval(),
        shutdown.subscribe(),
    ));

    let metrics = if args.metrics.allow_metrics {
        // One reading is taken up front so the first scrape has something to
        // compare against rather than reporting a utilisation of zero.
        service.sample_resources().await;
        let sampler = tokio::spawn(sample_resources(
            Arc::clone(&service),
            args.metrics.interval(),
            shutdown.subscribe(),
        ));
        info!(
            interval_ms = args.metrics.metrics_interval_ms,
            "serving Prometheus metrics at /metrics"
        );
        Some((sampler, Arc::clone(&service) as Arc<dyn MetricsSource>))
    } else {
        info!("metrics reporting is off");
        None
    };
    let (metrics_sampler, metrics_source) = match metrics {
        Some((sampler, source)) => (Some(sampler), Some(source)),
        None => (None, None),
    };

    let memory_guard = guard.map(|guard| {
        tokio::spawn(guard_memory(
            Arc::clone(&service),
            guard,
            // Checked more often than the mark is likely to be crossed, and
            // far less often than a request arrives: reading
            // /proc/self/status is cheap, purging is not, and the cooldown is
            // what bounds the purging.
            std::time::Duration::from_secs(2),
            shutdown.subscribe(),
        ))
    });

    // An identity check is configuration, not code: the transport takes
    // whatever implements the trait, and the binary picks one.
    let authenticator: Arc<dyn Authenticator> = match &args.auth.auth_token {
        Some(token) => {
            info!(
                rest = %args.rest_address,
                grpc = %args.grpc_address,
                "requiring a bearer token on both transports; health and \
                 metrics stay open"
            );
            Arc::new(StaticTokenAuth::new(token.clone()))
        }
        None => {
            info!("no authentication; every caller is served");
            Arc::new(NoAuth)
        }
    };

    let rest = tokio::spawn(rest_server::serve_with_auth(
        args.rest_address,
        Arc::clone(&service) as Arc<dyn Inference>,
        metrics_source,
        Arc::clone(&authenticator),
        args.listen_backlog,
        wait_for(shutdown.subscribe()),
    ));
    let grpc = tokio::spawn(grpc_server::serve_with_auth(
        args.grpc_address,
        Arc::clone(&service) as Arc<dyn Inference>,
        Arc::clone(&authenticator),
        args.listen_backlog,
        wait_for(shutdown.subscribe()),
    ));

    wait_for_signal().await;
    info!("shutting down");
    let _ = shutdown.send(());

    let (rest, grpc) = tokio::join!(rest, grpc);
    service.shutdown().await;
    sweeper.abort();
    if let Some(guard) = memory_guard {
        guard.abort();
    }
    if let Some(sampler) = metrics_sampler {
        sampler.abort();
    }

    // Report both outcomes: one transport failing must not hide the other.
    let rest = rest.context("the REST task panicked")?;
    let grpc = grpc.context("the gRPC task panicked")?;
    if let Err(e) = &rest {
        warn!(error = %e, "REST server stopped with an error");
    }
    if let Err(e) = &grpc {
        warn!(error = %e, "gRPC server stopped with an error");
    }
    rest.map_err(|e| anyhow::anyhow!("REST server failed: {e}"))?;
    grpc.map_err(|e| anyhow::anyhow!("gRPC server failed: {e}"))?;

    info!("stopped");
    Ok(())
}

/// Unloads models that have gone unused, until shutdown.
async fn sweep_cache(
    service: Arc<InferenceService>,
    interval: std::time::Duration,
    mut shutdown: broadcast::Receiver<()>,
) {
    let mut ticker = tokio::time::interval(interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    loop {
        tokio::select! {
            _ = ticker.tick() => {
                let unloaded = service.sweep_cache().await;
                if unloaded > 0 {
                    info!(unloaded, "swept idle models from the cache");
                }
            }
            _ = shutdown.recv() => return,
        }
    }
}

/// Re-reads CPU and memory use on the metrics interval, until shutdown.
async fn sample_resources(
    service: Arc<InferenceService>,
    interval: std::time::Duration,
    mut shutdown: broadcast::Receiver<()>,
) {
    let mut ticker = tokio::time::interval(interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    loop {
        tokio::select! {
            _ = ticker.tick() => { service.sample_resources().await; }
            _ = shutdown.recv() => return,
        }
    }
}

/// Watches resident memory and reclaims above the configured mark.
///
/// Runs on its own interval rather than inside the metrics sampler: a
/// deployment may turn metrics off and still want its memory limit honoured,
/// and a reclaim is a blocking syscall per region, which does not belong on
/// the thread that answers a scrape.
async fn guard_memory(
    service: Arc<InferenceService>,
    guard: Arc<galemind_service::MemoryGuard>,
    interval: std::time::Duration,
    mut shutdown: broadcast::Receiver<()>,
) {
    let mut ticker = tokio::time::interval(interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    loop {
        tokio::select! {
            _ = ticker.tick() => {
                // The purge is a syscall per region, so it goes to the
                // blocking pool; the unload is async, so it is decided here
                // and run after. Hence the two-step rather than a closure
                // that awaits.
                let decide = {
                    let guard = Arc::clone(&guard);
                    tokio::task::spawn_blocking(move || guard.check())
                };
                let outcome = match decide.await {
                    Ok(outcome) => outcome,
                    Err(e) => {
                        warn!("the memory guard panicked: {e}");
                        continue;
                    }
                };
                // Unloading is the guard's advice and this task's job: the
                // cache is async and the guard is not.
                if matches!(outcome, galemind_service::Reclaimed::StillOver { .. }) {
                    let unloaded = service.unload_all().await;
                    guard.record_unload(unloaded);
                    if unloaded > 0 {
                        warn!(unloaded, "unloaded models to get under the memory mark");
                    }
                }
            }
            _ = shutdown.recv() => return,
        }
    }
}

async fn wait_for(mut shutdown: broadcast::Receiver<()>) {
    let _ = shutdown.recv().await;
}

/// Resolves on Ctrl-C or on `SIGTERM`, which is how a container asks a process
/// to stop.
async fn wait_for_signal() {
    #[cfg(unix)]
    {
        let mut terminate = match signal::unix::signal(signal::unix::SignalKind::terminate()) {
            Ok(signal) => signal,
            Err(e) => {
                warn!(error = %e, "cannot listen for SIGTERM; Ctrl-C only");
                let _ = signal::ctrl_c().await;
                return;
            }
        };
        tokio::select! {
            _ = signal::ctrl_c() => info!("received Ctrl-C"),
            _ = terminate.recv() => info!("received SIGTERM"),
        }
    }
    #[cfg(not(unix))]
    {
        let _ = signal::ctrl_c().await;
    }
}

async fn list_models(args: RepositoryArgs) -> Result<()> {
    let store = FsModelStore::new(&args.model_repository);
    let keys = store
        .list()
        .await
        .map_err(anyhow::Error::from)
        .with_context(|| {
            format!(
                "cannot read the model repository {}",
                args.model_repository.display()
            )
        })?;

    if keys.is_empty() {
        println!("no models in {}", args.model_repository.display());
        return Ok(());
    }
    for key in keys {
        let artifact = store.fetch(&key).await.map_err(anyhow::Error::from)?;
        println!(
            "{:<28} {:<14} {:>10} bytes  max_batch_size={}",
            key.to_string(),
            artifact.platform,
            artifact.bytes.len(),
            artifact
                .config
                .max_batch_size
                .map_or("-".to_string(), |n| n.to_string())
        );
    }
    Ok(())
}

async fn ingest(args: RepositoryArgs) -> Result<()> {
    #[cfg(feature = "rocksdb-store")]
    {
        use galemind_store::RocksModelStore;

        if let Some(parent) = args.store_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let store = RocksModelStore::open(&args.store_path)
            .with_context(|| format!("cannot open {}", args.store_path.display()))?;
        let report = galemind_service::ingest_repository(&args.model_repository, &store)
            .await
            .map_err(anyhow::Error::from)?;

        println!(
            "added {}, updated {}, unchanged {}, failed {} -> {}",
            report.added,
            report.updated,
            report.unchanged,
            report.failed,
            args.store_path.display()
        );
        Ok(())
    }
    #[cfg(not(feature = "rocksdb-store"))]
    {
        let _ = args;
        anyhow::bail!("this build has no RocksDB support")
    }
}

#[cfg(all(test, feature = "jemalloc"))]
mod jemalloc_tests {
    use tikv_jemalloc_ctl::{Access, AsName};

    /// Which mallctl spelling jemalloc actually accepts for a forced purge.
    ///
    /// Written as a test because it is a property of the linked jemalloc
    /// rather than of this code: a version that stops accepting the form
    /// used here would otherwise turn the memory guard into a no-op that
    /// still logs success.
    #[test]
    fn the_purge_mallctl_is_accepted() {
        let narenas: u32 = b"arenas.narenas\0"
            .name()
            .read()
            .expect("jemalloc reports its arena count");
        assert!(narenas > 0);
        println!("narenas = {narenas}");

        // The "every arena" index, which the first attempt used.
        let all = b"arena.4096.dirty_decay_ms\0".name().write(0_isize);
        println!("arena.4096.dirty_decay_ms = {all:?}");

        // Per-arena, which is what the documentation describes as writable.
        let mut ok = 0;
        for i in 0..narenas {
            let name = format!("arena.{i}.dirty_decay_ms\0");
            if name.as_bytes().name().write(0_isize).is_ok() {
                ok += 1;
            }
        }
        println!("per-arena writes accepted: {ok} of {narenas}");

        // The server runs with the background thread on, which is the only
        // other difference between this test and the running server.
        let started = tikv_jemalloc_ctl::background_thread::write(true);
        println!("background_thread := true -> {started:?}");
        let with_bg = b"arena.4096.dirty_decay_ms\0".name().write(0_isize);
        println!("arena.4096.dirty_decay_ms with background thread = {with_bg:?}");
        let restore = b"arena.4096.dirty_decay_ms\0".name().write(60_000_isize);
        println!("restoring the window = {restore:?}");
        assert!(
            ok > 0 || all.is_ok(),
            "no spelling of a forced purge was accepted"
        );
    }
}
