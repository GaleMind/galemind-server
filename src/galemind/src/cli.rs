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

    /// Threads ONNX Runtime may use inside one operator, per model.
    ///
    /// Left unset it is derived from the machine and the repository: the
    /// cores are divided between the models that can be loaded at once, up
    /// to half the machine for a single model. See
    /// [`ThreadArgs::resolved_intra_op_threads`].
    ///
    /// `0` means something different and is kept: let ONNX Runtime choose,
    /// which is one thread per core for *every* loaded model -- 120 threads
    /// on a twelve-core machine with ten models.
    ///
    /// A model's own `config.json` may raise this for itself, which is how
    /// one heavy model gets more without the rest of the repository taking
    /// it too.
    #[arg(long, env = "GALEMIND_INTRA_OP_THREADS", value_name = "N")]
    pub intra_op_threads: Option<usize>,

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

    /// Operator threads per model, derived from the machine when not given.
    ///
    /// `loadable` is how many model versions can be in memory at once --
    /// the smaller of the cache's capacity and what the repository holds.
    /// That is the number operator threads multiply against, because ONNX
    /// Runtime's threads belong to a session: eight threads is eight threads
    /// with one model loaded and eighty with ten.
    ///
    /// So the cores are divided between them, with two floors and a ceiling:
    ///
    /// * at least one, since a session needs a thread to run on;
    /// * at most half the machine, so that a single model cannot take a box
    ///   whose other half is serving requests, decoding JSON and answering
    ///   scrapes -- which is where a quarter to a third of this server's CPU
    ///   time goes under load;
    /// * and the division rounds down, because an operator thread that has
    ///   to share a core with another session's costs more in contention
    ///   than it returns.
    ///
    /// On a twelve-core machine that is six threads for a single model, three
    /// for four models and one for ten; on a two-core machine it is one
    /// whatever is loaded. Measured, both ends matter: one model went from
    /// 200 to 354 requests a second between one thread and eight, and six
    /// models at once went the other way, 26 337 requests a second at one
    /// thread against 24 045 at six.
    pub fn resolved_intra_op_threads(&self, loadable: usize) -> usize {
        if let Some(threads) = self.intra_op_threads {
            // Including zero, which means "let ONNX Runtime decide".
            return threads;
        }
        let cores = Self::cpu_count();
        let half = (cores / 2).max(1);
        (cores / loadable.max(1)).clamp(1, half)
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

    /// Resident memory, in megabytes, above which the server reclaims.
    ///
    /// It asks the allocator for its retained pages first, which loses
    /// nothing -- they are pages already freed, kept in case the work came
    /// back -- and unloads the models only if that was not enough. Zero turns
    /// it off.
    ///
    /// Returning pages is not free: the next allocation faults them back in.
    /// So this is a limit to respect, not a target to sit at, and nothing
    /// happens below it.
    #[arg(
        long,
        env = "GALEMIND_MEMORY_HIGH_WATER_MB",
        default_value_t = 500,
        value_name = "MB"
    )]
    pub memory_high_water_mb: u64,

    /// Seconds to wait after a reclaim before another may run.
    ///
    /// A server whose steady state is over the mark would otherwise purge on
    /// every sample and spend itself faulting the same pages straight back
    /// in.
    #[arg(
        long,
        env = "GALEMIND_MEMORY_RECLAIM_COOLDOWN_SECS",
        default_value_t = 30,
        value_name = "SECS"
    )]
    pub memory_reclaim_cooldown_secs: u64,
}

impl MetricsArgs {
    pub fn interval(&self) -> Duration {
        Duration::from_millis(self.metrics_interval_ms.max(100))
    }

    /// The configured high-water mark in bytes, or `None` when it is off.
    pub fn memory_high_water_bytes(&self) -> Option<u64> {
        (self.memory_high_water_mb > 0).then(|| self.memory_high_water_mb * 1024 * 1024)
    }

    pub fn reclaim_cooldown(&self) -> Duration {
        Duration::from_secs(self.memory_reclaim_cooldown_secs)
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
    /// Largest batch the server will form, whatever the traffic and whatever
    /// a model's own `max_batch_size` says.
    ///
    /// Ten, because the size of an execution is what sets resident memory:
    /// one model's intermediates at a batch of ten are the difference between
    /// a plateau of 180 MB and one of 254 MB, measured. A model that wants
    /// less says so in its `config.json`, which is still honoured -- this is
    /// a ceiling, not a target -- and a deployment that wants more raises it
    /// with `GALEMIND_MAX_BATCH_SIZE`, which is what a throughput-bound
    /// server with one small model should do.
    #[arg(long, env = "GALEMIND_MAX_BATCH_SIZE", default_value_t = 10)]
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

    /// Batches of one model that may be executing at the same time.
    ///
    /// The dispatcher pipelines regardless of this: it forms the next batch
    /// while the current one runs, so a batch window is paid in parallel
    /// with an execution rather than after it.
    ///
    /// Leave it at one unless the backend can run a model concurrently.
    /// ONNX Runtime cannot -- its `Run` is not thread-safe, so a session
    /// sits behind a mutex and a second execution only waits on it, holding
    /// a blocking thread another model could use. Measured, raising it to
    /// four cost about a fifth of the throughput. To make one model use
    /// several cores, raise `--intra-op-threads`, which parallelises inside
    /// one execution.
    #[arg(
        long,
        env = "GALEMIND_MAX_CONCURRENT_BATCHES",
        default_value_t = 1,
        value_name = "N"
    )]
    pub max_concurrent_batches: usize,
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
        assert_eq!(args.batching.max_batch_size, 10);
        // One, because ONNX Runtime serialises a session's own executions;
        // the dispatcher pipelines rather than parallelising them.
        assert_eq!(args.batching.max_concurrent_batches, 1);
        assert_eq!(args.metrics.memory_high_water_mb, 500);
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
            max_concurrent_batches: 4,
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
    /// The operator-thread default divides the machine between the models
    /// that can be loaded at once, which is the number session threads
    /// multiply against.
    #[test]
    fn operator_threads_are_divided_between_the_models_that_can_be_loaded() {
        let derived = ThreadArgs {
            worker_threads: 0,
            blocking_threads: 0,
            intra_op_threads: None,
            inter_op_threads: 0,
            parallel_execution: false,
            cpu_mem_arena: false,
        };
        let cores = ThreadArgs::cpu_count();
        let half = (cores / 2).max(1);

        // One model may have half the machine, and no more: the other half
        // is serving requests.
        assert_eq!(derived.resolved_intra_op_threads(1), half);
        // Ten models on anything up to twenty cores get one each.
        assert_eq!(
            derived.resolved_intra_op_threads(10),
            (cores / 10).clamp(1, half)
        );
        // Never zero, however many models are loadable, since a session
        // needs a thread to run on.
        assert_eq!(derived.resolved_intra_op_threads(1_000), 1);
        assert_eq!(
            derived.resolved_intra_op_threads(0),
            half,
            "no models is not no cores"
        );

        // It divides, so more models never means more threads each.
        let mut previous = usize::MAX;
        for loadable in 1..=16 {
            let threads = derived.resolved_intra_op_threads(loadable);
            assert!(threads <= previous, "{loadable} models got more threads");
            assert!(threads >= 1);
            previous = threads;
        }
    }

    /// An explicit value is honoured, including the zero that means "let
    /// ONNX Runtime decide".
    #[test]
    fn an_explicit_operator_thread_count_wins_over_the_heuristic() {
        let explicit = |threads: Option<usize>| ThreadArgs {
            worker_threads: 0,
            blocking_threads: 0,
            intra_op_threads: threads,
            inter_op_threads: 0,
            parallel_execution: false,
            cpu_mem_arena: false,
        };

        assert_eq!(explicit(Some(3)).resolved_intra_op_threads(10), 3);
        assert_eq!(
            explicit(Some(0)).resolved_intra_op_threads(1),
            0,
            "zero is ONNX Runtime's own default, not an unset value"
        );
    }

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
        // Unset means derived, not one.
        assert_eq!(args.threads.intra_op_threads, None);
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
        assert_eq!(args.threads.intra_op_threads, Some(2));
        assert!(args.threads.cpu_mem_arena);
    }

    /// A blocking pool of zero threads cannot run an inference, so an
    /// explicit zero means the default rather than nothing.
    #[test]
    fn the_blocking_pool_always_has_at_least_one_thread() {
        let threads = |blocking| ThreadArgs {
            worker_threads: 0,
            blocking_threads: blocking,
            intra_op_threads: None,
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
            memory_high_water_mb: 0,
            memory_reclaim_cooldown_secs: 30,
        };
        assert_eq!(metrics.interval(), Duration::from_millis(100));
    }

    #[test]
    fn a_bad_address_is_refused_before_anything_starts() {
        assert!(Cli::try_parse_from(["galemind", "serve", "--rest-address", "nope"]).is_err());
    }
}
