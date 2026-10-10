//! The GaleMind server binary.

#![forbid(unsafe_code)]

mod cli;

use anyhow::{Context, Result};
use clap::Parser;
use cli::{Cli, Command, RepositoryArgs, ServeArgs, StoreKind};
use galemind_batching::{AdaptiveBatchConfig, AdaptiveBatchPolicy, DispatcherConfig};
use galemind_cache::LruTtlConfig;
use galemind_core::{BackendRegistry, Inference, ModelStore, SystemClock};
use galemind_onnx::OnnxBackendFactory;
use galemind_service::{InferenceService, ServiceBuilder};
use galemind_store::FsModelStore;
use std::sync::Arc;
use tokio::signal;
use tokio::sync::broadcast;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new(&cli.log))
        .with_target(true)
        .init();

    match cli.command {
        Command::Serve(args) => serve(args).await,
        Command::Models(args) => list_models(args).await,
        Command::Ingest(args) => ingest(args).await,
    }
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

async fn build_service(args: &ServeArgs) -> Result<Arc<InferenceService>> {
    let (store, usage) = build_store(&args.repository).await?;

    let backends =
        BackendRegistry::new().with(Arc::new(OnnxBackendFactory::new(args.intra_op_threads)));

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
        });
    if let Some(usage) = usage {
        builder = builder.with_usage_index(usage);
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

    Ok(service)
}

async fn serve(args: ServeArgs) -> Result<()> {
    let service = build_service(&args).await?;

    // One shutdown signal fans out to every task, so Ctrl-C drains the queues
    // instead of cutting requests off mid-flight.
    let (shutdown, _) = broadcast::channel::<()>(1);
    let sweeper = tokio::spawn(sweep_cache(
        Arc::clone(&service),
        args.cache.sweep_interval(),
        shutdown.subscribe(),
    ));

    let rest = tokio::spawn(rest_server::serve(
        args.rest_address,
        Arc::clone(&service) as Arc<dyn Inference>,
        wait_for(shutdown.subscribe()),
    ));
    let grpc = tokio::spawn(grpc_server::serve(
        args.grpc_address,
        Arc::clone(&service) as Arc<dyn Inference>,
        wait_for(shutdown.subscribe()),
    ));

    wait_for_signal().await;
    info!("shutting down");
    let _ = shutdown.send(());

    let (rest, grpc) = tokio::join!(rest, grpc);
    service.shutdown().await;
    sweeper.abort();

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
