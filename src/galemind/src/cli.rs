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
    /// `galemind=debug,rest_server=debug`).
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

    /// Connections the kernel may queue for each port before refusing more.
    ///
    /// The accept queue, not a connection limit: it bounds how many
    /// completed handshakes may wait to be accepted, which is what a burst
    /// of clients connecting at once fills. The kernel caps it at
    /// `net.core.somaxconn`, so raising this past that needs the sysctl too.
    #[arg(
        long,
        env = "GALEMIND_LISTEN_BACKLOG",
        default_value_t = galemind_service::DEFAULT_LISTEN_BACKLOG,
        value_name = "N"
    )]
    pub listen_backlog: u32,

    /// Load every model before reporting the server ready.
    #[arg(long, env = "GALEMIND_PRELOAD")]
    pub preload: bool,

    #[command(flatten)]
    pub cache: CacheArgs,

    #[command(flatten)]
    pub batching: BatchingArgs,

    #[command(flatten)]
    pub threads: ThreadArgs,

    #[command(flatten)]
    pub auth: AuthArgs,

    #[command(flatten)]
    pub metrics: MetricsArgs,
}

/// Who may call the protocol's endpoints.
///
/// The server ships open, because the identity provider is a deployment's
/// choice and the common case is a server reachable only from inside its own
/// network. A shared secret is the one check that needs no provider, so it is
/// the one the binary can offer; anything issuing real identities --
/// Keycloak, or another OIDC provider -- implements
/// `rest_server::Authenticator` and is wired in here, which is a line of
/// code rather than a change to the transport.
///
/// Health probes and the metrics scrape are never authenticated; see
/// `rest_server::auth`.
#[derive(Debug, Args, Clone)]
pub struct AuthArgs {
    /// Require this bearer token on every protocol request. Unset leaves the
    /// server open.
    ///
    /// Prefer the environment variable: an argument is visible in the process
    /// list to every user on the machine.
    #[arg(long, env = "GALEMIND_AUTH_TOKEN", value_name = "TOKEN")]
    pub auth_token: Option<String>,
}

/// How the server is allowed to use the machine's cores.
///
/// There are two thread pools and they multiply. Requests are served on async
/// worker threads, and each model's batches execute on a blocking thread,
/// where ONNX Runtime may use `intra_op_threads` more inside a single
/// operator. Ten models being served at once with ONNX Runtime left to choose
/// for itself is ten thread pools of one-per-core: on a 12-core machine, 120
/// threads fighting over 12 cores, which costs more in contention than it
/// gains in parallelism.
///
/// The defaults therefore take the parallelism from serving models and
/// batches side by side -- one worker per core, one thread per operator --
/// which is the right shape for many small CPU models. A server with one
/// large model and spare cores is the opposite case, and wants
/// `--intra-op-threads` raised to match.
#[derive(Debug, Args, Clone, Copy)]
pub struct ThreadArgs {
    /// Async worker threads. 0 means one per core.
    #[arg(long, env = "GALEMIND_WORKER_THREADS", default_value_t = 0)]
    pub worker_threads: usize,

    /// Threads for blocking work: one per batch being executed, plus model
    /// loads and store reads. 0 means twice the core count, with a floor of
    /// four.
    ///
    /// This bounds how many batches can execute at once, so it is also the
    /// ceiling on how much of the machine inference can take. Tokio's own
    /// default is 512, which would let a burst across many models start 512
    /// inferences at once on a 12-core machine.
    #[arg(long, env = "GALEMIND_BLOCKING_THREADS", default_value_t = 0)]
    pub blocking_threads: usize,

    /// Threads ONNX Runtime may use inside one operator, per model. 0 lets
    /// ONNX Runtime choose, which is one per core for every loaded model.
    ///
    /// A model's own `config.json` may raise this for itself.
    #[arg(long, env = "GALEMIND_INTRA_OP_THREADS", default_value_t = 1)]
    pub intra_op_threads: usize,

    /// Run independent branches of a model's graph at once. Worth it for a
    /// wide graph such as a large ensemble, pointless for a deep chain, and it
    /// adds a second ONNX Runtime thread pool per model.
    #[arg(long, env = "GALEMIND_PARALLEL_EXECUTION")]
    pub parallel_execution: bool,

    /// Threads ONNX Runtime may use to run independent operators at once; only
    /// used with `--parallel-execution`. 0 lets ONNX Runtime choose.
    #[arg(long, env = "GALEMIND_INTER_OP_THREADS", default_value_t = 0)]
    pub inter_op_threads: usize,

    /// Keep ONNX Runtime's CPU arena allocator.
    ///
    /// The arena speeds up repeated runs of the same shape by never giving
    /// memory back: resident memory settles at the high-water mark of the
    /// largest batch each loaded model has ever seen. Off by default, because
    /// a server with many models and a varying batch size pays that as
    /// unbounded-looking growth. Turn it on for a single model under steady
    /// load, where the high-water mark is reached quickly and stays.
    #[arg(long, env = "GALEMIND_CPU_MEM_ARENA")]
    pub cpu_mem_arena: bool,
}

impl ThreadArgs {
    /// Cores the process may use.
    pub fn cpu_count() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }

    /// Async worker threads to start, resolving 0 to one per core.
    pub fn resolved_worker_threads(&self) -> usize {
        match self.worker_threads {
            0 => Self::cpu_count(),
            n => n,
        }
    }

    /// Blocking threads to allow, resolving 0 to twice the core count.
    pub fn resolved_blocking_threads(&self) -> usize {
        match self.blocking_threads {
            0 => (Self::cpu_count() * 2).max(4),
            n => n.max(1),
        }
    }
}

/// Prometheus metrics reporting.
#[derive(Debug, Args)]
pub struct MetricsArgs {
    /// Serve Prometheus metrics at /metrics on the HTTP port. Pass
    /// `--allow-metrics false` to expose nothing but the protocol.
    #[arg(
        long,
        env = "GALEMIND_ALLOW_METRICS",
        default_value_t = true,
        action = clap::ArgAction::Set,
        value_name = "BOOL"
    )]
    pub allow_metrics: bool,

    /// How often to re-read CPU and memory use. Utilisation is a rate, so it
    /// is measured between readings rather than during a scrape; request
    /// counters are unaffected and update per request.
    #[arg(long, env = "GALEMIND_METRICS_INTERVAL_MS", default_value_t = 1_000)]
    pub metrics_interval_ms: u64,
}

impl MetricsArgs {
    pub fn interval(&self) -> Duration {
        Duration::from_millis(self.metrics_interval_ms.max(100))
    }
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
        assert_eq!(
            args.listen_backlog,
            galemind_service::DEFAULT_LISTEN_BACKLOG
        );
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

    /// The server ships open; a token is opt-in.
    #[test]
    fn there_is_no_authentication_unless_a_token_is_configured() {
        let cli = Cli::try_parse_from(["galemind", "serve"]).unwrap();
        let Command::Serve(args) = cli.command else {
            panic!("expected serve");
        };
        assert_eq!(args.auth.auth_token, None);

        let cli = Cli::try_parse_from(["galemind", "serve", "--auth-token", "s3cret"]).unwrap();
        let Command::Serve(args) = cli.command else {
            panic!("expected serve");
        };
        assert_eq!(args.auth.auth_token.as_deref(), Some("s3cret"));
    }

    #[test]
    fn metrics_are_served_by_default_and_can_be_turned_off() {
        let cli = Cli::try_parse_from(["galemind", "serve"]).unwrap();
        let Command::Serve(args) = cli.command else {
            panic!("expected serve");
        };
        assert!(args.metrics.allow_metrics);
        assert_eq!(args.metrics.interval(), Duration::from_secs(1));

        let cli = Cli::try_parse_from(["galemind", "serve", "--allow-metrics", "false"]).unwrap();
        let Command::Serve(args) = cli.command else {
            panic!("expected serve");
        };
        assert!(!args.metrics.allow_metrics);
    }

    /// Zero means "one per core" for workers and "twice that" for blocking
    /// work, so the defaults scale with the machine instead of pinning a
    /// figure chosen on another one.
    #[test]
    fn the_thread_plan_defaults_to_the_machines_core_count() {
        let cli = Cli::try_parse_from(["galemind", "serve"]).unwrap();
        let Command::Serve(args) = cli.command else {
            panic!("expected serve");
        };
        let cores = ThreadArgs::cpu_count();

        assert_eq!(args.threads.worker_threads, 0, "0 means one per core");
        assert_eq!(args.threads.resolved_worker_threads(), cores);
        assert_eq!(args.threads.resolved_blocking_threads(), (cores * 2).max(4));
        // One operator thread per model, not one per core per model.
        assert_eq!(args.threads.intra_op_threads, 1);
        assert!(!args.threads.parallel_execution);
        // The arena is what makes resident memory climb and never fall.
        assert!(!args.threads.cpu_mem_arena);
    }

    #[test]
    fn the_thread_plan_can_be_pinned_explicitly() {
        let cli = Cli::try_parse_from([
            "galemind",
            "serve",
            "--worker-threads",
            "4",
            "--blocking-threads",
            "8",
            "--intra-op-threads",
            "2",
            "--cpu-mem-arena",
        ])
        .unwrap();
        let Command::Serve(args) = cli.command else {
            panic!("expected serve");
        };
        assert_eq!(args.threads.resolved_worker_threads(), 4);
        assert_eq!(args.threads.resolved_blocking_threads(), 8);
        assert_eq!(args.threads.intra_op_threads, 2);
        assert!(args.threads.cpu_mem_arena);
    }

    /// A blocking pool of zero threads cannot run an inference, so an
    /// explicit zero means the default rather than nothing.
    #[test]
    fn the_blocking_pool_always_has_at_least_one_thread() {
        let threads = |blocking| ThreadArgs {
            worker_threads: 0,
            blocking_threads: blocking,
            intra_op_threads: 1,
            parallel_execution: false,
            inter_op_threads: 0,
            cpu_mem_arena: false,
        };
        assert!(threads(0).resolved_blocking_threads() >= 4);
        assert_eq!(threads(1).resolved_blocking_threads(), 1);
    }

    #[test]
    fn the_metrics_interval_has_a_floor_so_sampling_cannot_spin() {
        let metrics = MetricsArgs {
            allow_metrics: true,
            metrics_interval_ms: 0,
        };
        assert_eq!(metrics.interval(), Duration::from_millis(100));
    }

    #[test]
    fn a_bad_address_is_refused_before_anything_starts() {
        assert!(Cli::try_parse_from(["galemind", "serve", "--rest-address", "nope"]).is_err());
    }
}
