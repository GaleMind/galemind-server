//! What this process is and how it is allowed to use the machine.
//!
//! Reported in the metrics and logged once at startup. A server that is slow
//! because it was given one worker thread, or because every model opened a
//! thread pool per core, looks exactly like a server that is slow because the
//! model is slow -- unless the thread plan is written down where a scrape can
//! see it.

/// The thread plan and build identity of a running server.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeInfo {
    pub version: &'static str,
    /// Architecture, in the naming operators use for images and packages:
    /// `amd64`, `arm64`.
    pub platform: &'static str,
    pub os: &'static str,
    /// Cores the process may use.
    pub cpu_count: usize,
    /// Async worker threads. Requests are decoded, queued and answered on
    /// these.
    pub worker_threads: usize,
    /// Threads available for blocking work. A batch executes on one of these,
    /// so this bounds how many inferences can run at once.
    pub blocking_threads: usize,
    /// Threads ONNX Runtime may use inside one operator, per session.
    pub intra_op_threads: usize,
    /// Model versions that can be in memory at once: the smaller of the
    /// cache's capacity and what the repository holds.
    ///
    /// Part of the thread plan because operator threads belong to a session,
    /// so this is what they multiply against -- and because one model's
    /// executions cannot overlap (ONNX Runtime's `Run` is not thread-safe),
    /// it also bounds how many executions can be in flight at once. Zero
    /// where it is not known, which is what a default [`RuntimeInfo`]
    /// reports.
    pub loadable_models: usize,
    /// Whether ONNX Runtime's CPU arena allocator is in use.
    pub cpu_mem_arena: bool,
    /// `MALLOC_ARENA_MAX`, when the environment sets it.
    ///
    /// Reported because it is the single largest influence on this server's
    /// resident memory and it cannot be set from inside the process: glibc
    /// reads it before `main` runs. Left unset, glibc allows up to eight
    /// arenas per core -- 96 on a twelve-core machine -- and each keeps the
    /// memory it has freed. Inference frees in large blocks from many
    /// threads, so that retention is most of the difference between a server
    /// that settles at 70 MB and one that settles at 200 MB.
    pub malloc_arena_max: Option<usize>,
    /// The memory allocator the binary was linked against.
    ///
    /// Reported because it decides whether an idle server gives its memory
    /// back. glibc keeps what it has freed in per-thread arenas and has no
    /// way to return it short of `malloc_trim`; jemalloc returns unused pages
    /// on a decay timer. Measured on this repository, after driving all ten
    /// models and then standing idle: glibc held 118 MB indefinitely, and
    /// jemalloc with a 60-second decay fell from 214 MB to 87 MB with every
    /// model still loaded. An operator looking at a memory graph needs to
    /// know which of those two they are looking at.
    pub allocator: &'static str,
}

/// The name [`RuntimeInfo::detected`] reports when nobody says otherwise.
///
/// Whoever installs a `#[global_allocator]` is the only code that knows what
/// it installed, and that is the binary, not this crate. So this is the
/// honest default -- the platform's allocator -- and the binary overrides the
/// field when it has replaced it. Inferring it here from a feature of *this*
/// crate is what the first version did, and it reported `system` for a binary
/// demonstrably running jemalloc, because enabling a feature on a binary does
/// not enable the same-named feature on its dependencies.
pub const SYSTEM_ALLOCATOR: &str = "system";

impl RuntimeInfo {
    /// The build's identity, with a thread plan of one worker per core and
    /// ONNX Runtime left at this server's defaults.
    pub fn detected() -> Self {
        let cpu_count = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        Self {
            version: galemind_core::SERVER_VERSION,
            platform: platform(),
            os: std::env::consts::OS,
            cpu_count,
            worker_threads: cpu_count,
            blocking_threads: (cpu_count * 2).max(4),
            intra_op_threads: 1,
            loadable_models: 0,
            cpu_mem_arena: false,
            malloc_arena_max: malloc_arena_max(),
            allocator: SYSTEM_ALLOCATOR,
        }
    }

    /// Total threads that may run model code at once.
    ///
    /// Model code does not run on the async workers: a batch is handed to the
    /// blocking pool, and ONNX Runtime may use `intra_op_threads` inside it.
    /// So this is the executions that can be in flight times the operator
    /// threads each may use, and comparing it with `cpu_count` is what says
    /// whether the machine is oversubscribed.
    ///
    /// The executions in flight are bounded by the blocking pool *and* by
    /// how many models can be loaded, because ONNX Runtime's `Run` is not
    /// thread-safe: a session runs one batch at a time, so ten loaded models
    /// cannot have eleven executions between them however many blocking
    /// threads there are. Reporting the pool alone said 144 threads for a
    /// server with one model and six operator threads, which could use six.
    pub fn compute_threads(&self) -> usize {
        let executions = match self.loadable_models {
            // Unknown: the blocking pool is then the only bound there is.
            0 => self.blocking_threads,
            models => models.min(self.blocking_threads),
        };
        executions * self.intra_op_threads.max(1)
    }
}

/// Reads `MALLOC_ARENA_MAX`, which only the environment can set.
pub fn malloc_arena_max() -> Option<usize> {
    std::env::var("MALLOC_ARENA_MAX")
        .ok()?
        .trim()
        .parse::<usize>()
        .ok()
        .filter(|value| *value > 0)
}

/// `std::env::consts::ARCH` under the names used for container images.
///
/// Rust says `x86_64` where an image tag, a Debian package and an operator all
/// say `amd64`; reporting the Rust spelling would mean every dashboard
/// translating it back.
pub fn platform() -> &'static str {
    match std::env::consts::ARCH {
        "x86_64" => "amd64",
        "aarch64" => "arm64",
        "x86" => "386",
        "arm" => "arm",
        "powerpc64" => "ppc64le",
        "s390x" => "s390x",
        "riscv64" => "riscv64",
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What the formula is for, stated as a test: a thread plan that cannot
    /// be reached should not be reported as if it could.
    #[test]
    fn the_compute_ceiling_counts_executions_rather_than_threads() {
        let one_model = RuntimeInfo {
            blocking_threads: 24,
            intra_op_threads: 6,
            loadable_models: 1,
            ..RuntimeInfo::detected()
        };
        // One model runs one batch at a time, so six threads, not 144.
        assert_eq!(one_model.compute_threads(), 6);

        let zoo = RuntimeInfo {
            blocking_threads: 24,
            intra_op_threads: 1,
            loadable_models: 11,
            ..RuntimeInfo::detected()
        };
        assert_eq!(zoo.compute_threads(), 11);

        // More models than blocking threads: the pool is the bound again.
        let many = RuntimeInfo {
            blocking_threads: 4,
            intra_op_threads: 2,
            loadable_models: 100,
            ..RuntimeInfo::detected()
        };
        assert_eq!(many.compute_threads(), 8);

        // Unknown, which is what a bare report says: the pool alone.
        let unknown = RuntimeInfo {
            blocking_threads: 8,
            intra_op_threads: 2,
            loadable_models: 0,
            ..RuntimeInfo::detected()
        };
        assert_eq!(unknown.compute_threads(), 16);
    }

    /// The allocator has to be reported, because it is what decides whether
    /// an idle server's memory comes back -- and it has to come from whoever
    /// installed it, which is the binary.
    #[test]
    fn the_allocator_defaults_to_the_platforms_and_can_be_overridden() {
        assert_eq!(RuntimeInfo::detected().allocator, SYSTEM_ALLOCATOR);

        let replaced = RuntimeInfo {
            allocator: "jemalloc",
            ..RuntimeInfo::detected()
        };
        assert_eq!(replaced.allocator, "jemalloc");
    }

    /// Only a positive number means anything; anything else is reported as
    /// unset rather than as zero arenas.
    #[test]
    fn an_unparseable_arena_limit_reads_as_unset() {
        // The variable is process-wide, so the parsing is tested through the
        // same path with the value this process actually has.
        let parsed = malloc_arena_max();
        match std::env::var("MALLOC_ARENA_MAX") {
            Ok(value) => match value.trim().parse::<usize>() {
                Ok(n) if n > 0 => assert_eq!(parsed, Some(n)),
                _ => assert_eq!(parsed, None),
            },
            Err(_) => assert_eq!(parsed, None),
        }
    }

    #[test]
    fn the_architecture_is_reported_as_operators_name_it() {
        // Whatever this test runs on, the name must be one of the image-tag
        // spellings rather than the Rust one.
        assert!(
            !platform().contains("x86_64") && !platform().contains("aarch64"),
            "{} is the Rust spelling, not the image-tag one",
            platform()
        );
        assert!(!platform().is_empty());
    }

    #[test]
    fn the_detected_plan_is_one_worker_per_core_without_an_arena() {
        let info = RuntimeInfo::detected();
        assert_eq!(info.worker_threads, info.cpu_count);
        assert!(info.blocking_threads >= info.worker_threads);
        assert!(info.cpu_count >= 1);
        assert_eq!(info.intra_op_threads, 1);
        assert!(!info.cpu_mem_arena);
        assert_eq!(info.version, galemind_core::SERVER_VERSION);
    }

    /// The figure that says whether the machine is oversubscribed has to
    /// account for both kinds of thread, not just the async workers.
    #[test]
    fn compute_threads_multiply_workers_by_the_threads_each_operator_may_use() {
        // Model code runs on the blocking pool, so that is the pool the
        // operator threads multiply -- counting the async workers instead
        // would report the wrong figure whenever the two differ.
        let info = RuntimeInfo {
            worker_threads: 4,
            blocking_threads: 8,
            intra_op_threads: 4,
            ..RuntimeInfo::detected()
        };
        assert_eq!(info.compute_threads(), 32);

        // Zero means "let ONNX Runtime choose", which is at least one.
        let info = RuntimeInfo {
            worker_threads: 4,
            blocking_threads: 8,
            intra_op_threads: 0,
            ..RuntimeInfo::detected()
        };
        assert_eq!(info.compute_threads(), 8);
    }
}
