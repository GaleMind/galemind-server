//! The command line.

use clap::{Args, Parser, Subcommand, ValueEnum};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

/// A CPU inference server speaking the Predict Protocol v2.
#[derive(Debug, Parser)]
#[command(name = "galemind", version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,

    /// Log filter, in the syntax of `RUST_LOG` (for example `info` or
    /// `galemind=debug,tower_http=debug`).
    #[arg(long, global = true, env = "GALEMIND_LOG", default_value = "info")]
    pub log: String,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Serve the protocol over HTTP and gRPC.
    Serve(ServeArgs),
    /// List the model versions in the repository and exit.
    Models(RepositoryArgs),
    /// Copy the model repository into the durable store and exit.
    Ingest(RepositoryArgs),
}

/// Where models are read from and kept.
#[derive(Debug, Args, Clone)]
pub struct RepositoryArgs {
    /// Directory holding `<model>/<version>/model.onnx`.
    #[arg(
        long,
        env = "GALEMIND_MODEL_REPOSITORY",
        default_value = "models",
        value_name = "DIR"
    )]
    pub model_repository: PathBuf,

    /// Where the durable model store lives.
    #[arg(
        long,
        env = "GALEMIND_STORE_PATH",
        default_value = "data/models.db",
        value_name = "DIR"
    )]
    pub store_path: PathBuf,

    /// Which store serves models at request time.
    #[arg(long, env = "GALEMIND_STORE", default_value = "rocksdb")]
    pub store: StoreKind,
}

/// The store a running server reads models from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum StoreKind {
    /// Ingest the repository into RocksDB at startup and serve from there.
    /// Artifacts, metadata and usage records are then durable.
    Rocksdb,
    /// Read model files straight from the repository directory.
    Filesystem,
}

#[derive(Debug, Args)]
pub struct ServeArgs {
    #[command(flatten)]
    pub repository: RepositoryArgs,

    /// Address for the HTTP/REST endpoints.
    #[arg(
        long,
        env = "GALEMIND_REST_ADDRESS",
        default_value = "0.0.0.0:8000",
        value_name = "ADDR"
    )]
    pub rest_address: SocketAddr,

    /// Address for the gRPC endpoints.
    #[arg(
        long,
        env = "GALEMIND_GRPC_ADDRESS",
        default_value = "0.0.0.0:8001",
        value_name = "ADDR"
    )]
    pub grpc_address: SocketAddr,

    /// Load every model before reporting the server ready.
    #[arg(long, env = "GALEMIND_PRELOAD")]
    pub preload: bool,

    #[command(flatten)]
    pub cache: CacheArgs,

    #[command(flatten)]
    pub batching: BatchingArgs,

    /// Threads ONNX Runtime may use inside one operator; 0 lets it choose.
    #[arg(long, env = "GALEMIND_INTRA_OP_THREADS", default_value_t = 0)]
    pub intra_op_threads: usize,
}

/// How many models stay loaded, and for how long.
#[derive(Debug, Args)]
pub struct CacheArgs {
    /// Models kept loaded at once; the least recently used is unloaded first.
    #[arg(long, env = "GALEMIND_CACHE_CAPACITY", default_value_t = 8)]
    pub cache_capacity: usize,

    /// Unload a model after this many seconds without use.
    #[arg(long, env = "GALEMIND_CACHE_TTL_SECS", default_value_t = 3_600)]
    pub cache_ttl_secs: u64,

    /// How often to look for models that have outlived their time-to-live.
    #[arg(long, env = "GALEMIND_CACHE_SWEEP_SECS", default_value_t = 60)]
    pub cache_sweep_secs: u64,
}

/// Limits for the adaptive batcher.
#[derive(Debug, Args)]
pub struct BatchingArgs {
    /// Largest batch the server will form, whatever the traffic.
    #[arg(long, env = "GALEMIND_MAX_BATCH_SIZE", default_value_t = 32)]
    pub max_batch_size: usize,

    /// Shortest time to wait for a batch to fill.
    #[arg(long, env = "GALEMIND_MIN_BATCH_WINDOW_MICROS", default_value_t = 200)]
    pub min_batch_window_micros: u64,

    /// Longest time to wait for a batch to fill, which caps the latency
    /// batching can add.
    #[arg(
        long,
        env = "GALEMIND_MAX_BATCH_WINDOW_MICROS",
        default_value_t = 5_000
    )]
    pub max_batch_window_micros: u64,

    /// Requests that may wait per model before the server reports itself
    /// overloaded.
    #[arg(long, env = "GALEMIND_QUEUE_CAPACITY", default_value_t = 1_024)]
    pub queue_capacity: usize,

    /// Fail a request that has waited this long rather than running it late.
    #[arg(long, env = "GALEMIND_QUEUE_TIMEOUT_SECS", default_value_t = 10)]
    pub queue_timeout_secs: u64,
}

impl CacheArgs {
    pub fn ttl(&self) -> Duration {
        Duration::from_secs(self.cache_ttl_secs)
    }

    pub fn sweep_interval(&self) -> Duration {
        Duration::from_secs(self.cache_sweep_secs.max(1))
    }
}

impl BatchingArgs {
    pub fn min_window(&self) -> Duration {
        Duration::from_micros(self.min_batch_window_micros)
    }

    pub fn max_window(&self) -> Duration {
        Duration::from_micros(
            self.max_batch_window_micros
                .max(self.min_batch_window_micros),
        )
    }

    pub fn queue_timeout(&self) -> Duration {
        Duration::from_secs(self.queue_timeout_secs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn the_command_line_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn serving_needs_no_arguments_at_all() {
        let cli = Cli::try_parse_from(["galemind", "serve"]).unwrap();
        let Command::Serve(args) = cli.command else {
            panic!("expected serve");
        };
        assert_eq!(args.rest_address.port(), 8000);
        assert_eq!(args.grpc_address.port(), 8001);
        assert_eq!(args.repository.model_repository, PathBuf::from("models"));
        assert_eq!(args.repository.store, StoreKind::Rocksdb);
        assert_eq!(args.cache.cache_ttl_secs, 3_600);
        assert_eq!(args.batching.max_batch_size, 32);
        assert!(!args.preload);
    }

    #[test]
    fn addresses_and_limits_can_be_overridden() {
        let cli = Cli::try_parse_from([
            "galemind",
            "serve",
            "--rest-address",
            "127.0.0.1:9000",
            "--grpc-address",
            "127.0.0.1:9001",
            "--model-repository",
            "/srv/models",
            "--store",
            "filesystem",
            "--cache-capacity",
            "2",
            "--cache-ttl-secs",
            "60",
            "--max-batch-size",
            "4",
            "--preload",
        ])
        .unwrap();

        let Command::Serve(args) = cli.command else {
            panic!("expected serve");
        };
        assert_eq!(args.rest_address.to_string(), "127.0.0.1:9000");
        assert_eq!(
            args.repository.model_repository,
            PathBuf::from("/srv/models")
        );
        assert_eq!(args.repository.store, StoreKind::Filesystem);
        assert_eq!(args.cache.cache_capacity, 2);
        assert_eq!(args.cache.ttl(), Duration::from_secs(60));
        assert_eq!(args.batching.max_batch_size, 4);
        assert!(args.preload);
    }

    #[test]
    fn the_batch_window_cannot_be_inverted_by_configuration() {
        let batching = BatchingArgs {
            max_batch_size: 8,
            min_batch_window_micros: 5_000,
            max_batch_window_micros: 100,
            queue_capacity: 8,
            queue_timeout_secs: 1,
        };
        assert_eq!(batching.max_window(), batching.min_window());
    }

    #[test]
    fn a_bad_address_is_refused_before_anything_starts() {
        assert!(Cli::try_parse_from(["galemind", "serve", "--rest-address", "nope"]).is_err());
    }
}
