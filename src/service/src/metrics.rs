//! Prometheus metrics.
//!
//! The metric names follow NVIDIA Triton's, with `gm_` where Triton uses
//! `nv_`, so existing dashboards and alerts port across with a prefix change.
//! Metrics that cannot mean anything here are left out rather than reported as
//! zero: there are no GPU metrics on a CPU server, no pinned-memory metrics
//! without a GPU to pin for, and no response-cache metrics without a response
//! cache. What remains is request counts and latencies per model, the model
//! cache, this server's own batching, and process CPU and memory.
//!
//! Counters are cumulative and never reset, which is what Prometheus expects:
//! rates are computed by the scraper from the difference between scrapes.

use crate::resources::ResourceSnapshot;
use crate::runtime::RuntimeInfo;
use galemind_batching::DispatcherSnapshot;
use galemind_core::{CacheStats, InferError, ModelKey};
use std::collections::HashMap;
use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

/// Why a request failed, as a metric label.
///
/// Triton's categories are REJECTED, CANCELED, BACKEND and OTHER. Cancellation
/// cannot be observed here, and an invalid request is worth telling apart from
/// a backend fault because only one of them is the client's to fix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureReason {
    /// Refused because the queue was full.
    Rejected,
    /// Failed after waiting past the queue timeout.
    Timeout,
    /// The model or the runtime failed.
    Backend,
    /// The request was not valid for the model.
    Invalid,
    Other,
}

impl FailureReason {
    /// Every reason, so a scrape reports all series from the first request on
    /// and a dashboard has no gaps.
    pub const ALL: [FailureReason; 5] = [
        FailureReason::Rejected,
        FailureReason::Timeout,
        FailureReason::Backend,
        FailureReason::Invalid,
        FailureReason::Other,
    ];

    pub const fn label(&self) -> &'static str {
        match self {
            Self::Rejected => "rejected",
            Self::Timeout => "timeout",
            Self::Backend => "backend",
            Self::Invalid => "invalid",
            Self::Other => "other",
        }
    }

    fn index(&self) -> usize {
        match self {
            Self::Rejected => 0,
            Self::Timeout => 1,
            Self::Backend => 2,
            Self::Invalid => 3,
            Self::Other => 4,
        }
    }

    /// Classifies a failure.
    pub fn of(error: &InferError) -> Self {
        match error {
            InferError::Overloaded => Self::Rejected,
            InferError::Timeout(_) => Self::Timeout,
            InferError::Backend(_) | InferError::Store(_) => Self::Backend,
            InferError::InvalidRequest(_)
            | InferError::ModelNotFound(_)
            | InferError::VersionNotFound { .. }
            | InferError::Unsupported(_)
            | InferError::ModelNotReady(_) => Self::Invalid,
            InferError::Remote { client_error, .. } => {
                if *client_error {
                    Self::Invalid
                } else {
                    Self::Backend
                }
            }
            InferError::Internal(_) => Self::Other,
        }
    }
}

/// Upper bounds, in seconds, of the request-duration histogram.
///
/// They span a CPU model answering in well under a millisecond up to a request
/// that waited behind a full queue, with the most resolution where a tree
/// ensemble actually lands.
const DURATION_BUCKETS_SECS: [f64; 12] = [
    0.000_1, 0.000_25, 0.000_5, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0,
];

#[derive(Debug, Default)]
struct ModelCounters {
    success: AtomicU64,
    failures: [AtomicU64; 5],
    request_duration_micros: AtomicU64,
    histogram: [AtomicU64; DURATION_BUCKETS_SECS.len()],
    histogram_count: AtomicU64,
    histogram_sum_micros: AtomicU64,
}

impl ModelCounters {
    fn observe(&self, duration: Duration) {
        let micros = duration.as_micros() as u64;
        self.request_duration_micros
            .fetch_add(micros, Ordering::Relaxed);
        self.histogram_count.fetch_add(1, Ordering::Relaxed);
        self.histogram_sum_micros
            .fetch_add(micros, Ordering::Relaxed);

        // Buckets are cumulative in the exposition format, so this records the
        // first bound the observation falls under and rendering adds them up.
        let seconds = duration.as_secs_f64();
        if let Some(index) = DURATION_BUCKETS_SECS
            .iter()
            .position(|bound| seconds <= *bound)
        {
            self.histogram[index].fetch_add(1, Ordering::Relaxed);
        }
    }

    fn snapshot(&self) -> ModelMetricsSnapshot {
        ModelMetricsSnapshot {
            success: self.success.load(Ordering::Relaxed),
            failures: std::array::from_fn(|i| self.failures[i].load(Ordering::Relaxed)),
            request_duration_micros: self.request_duration_micros.load(Ordering::Relaxed),
            histogram: std::array::from_fn(|i| self.histogram[i].load(Ordering::Relaxed)),
            histogram_count: self.histogram_count.load(Ordering::Relaxed),
            histogram_sum_micros: self.histogram_sum_micros.load(Ordering::Relaxed),
        }
    }
}

/// What one model's request counters hold.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ModelMetricsSnapshot {
    pub success: u64,
    /// Indexed as [`FailureReason::ALL`].
    pub failures: [u64; 5],
    pub request_duration_micros: u64,
    /// Observations per bucket, not yet cumulative.
    pub histogram: [u64; DURATION_BUCKETS_SECS.len()],
    pub histogram_count: u64,
    pub histogram_sum_micros: u64,
}

impl ModelMetricsSnapshot {
    pub fn failures(&self, reason: FailureReason) -> u64 {
        self.failures[reason.index()]
    }

    pub fn total_failures(&self) -> u64 {
        self.failures.iter().sum()
    }
}

/// Per-model request counters.
///
/// The batching dispatchers already count batches and rows; this counts what
/// only the service sees, namely whether a request succeeded and how long it
/// took end to end.
#[derive(Default)]
pub struct MetricsRegistry {
    models: RwLock<HashMap<ModelKey, Arc<ModelCounters>>>,
    /// Failures of requests that never reached a model, such as a request for
    /// a model that does not exist.
    unresolved: [AtomicU64; 5],
}

impl MetricsRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    async fn counters(&self, key: &ModelKey) -> Arc<ModelCounters> {
        if let Some(counters) = self.models.read().await.get(key) {
            return Arc::clone(counters);
        }
        let mut models = self.models.write().await;
        Arc::clone(models.entry(key.clone()).or_default())
    }

    /// Records a request that was answered.
    pub async fn record_success(&self, key: &ModelKey, duration: Duration) {
        let counters = self.counters(key).await;
        counters.success.fetch_add(1, Ordering::Relaxed);
        counters.observe(duration);
    }

    /// Records a request that failed. `key` is `None` when the request never
    /// resolved to a model.
    pub async fn record_failure(
        &self,
        key: Option<&ModelKey>,
        error: &InferError,
        duration: Duration,
    ) {
        let reason = FailureReason::of(error);
        match key {
            Some(key) => {
                let counters = self.counters(key).await;
                counters.failures[reason.index()].fetch_add(1, Ordering::Relaxed);
                counters.observe(duration);
            }
            None => {
                self.unresolved[reason.index()].fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Every model's counters, sorted by model.
    pub async fn snapshot(&self) -> Vec<(ModelKey, ModelMetricsSnapshot)> {
        let models = self.models.read().await;
        let mut out: Vec<_> = models
            .iter()
            .map(|(key, counters)| (key.clone(), counters.snapshot()))
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// Failures of requests that never reached a model.
    pub fn unresolved_failures(&self) -> [u64; 5] {
        std::array::from_fn(|i| self.unresolved[i].load(Ordering::Relaxed))
    }
}

/// Everything a scrape needs, gathered before rendering.
pub struct PrometheusReport {
    pub server_version: &'static str,
    /// The thread plan and build identity this server is running with.
    pub runtime: RuntimeInfo,
    pub models: Vec<(ModelKey, ModelMetricsSnapshot)>,
    pub dispatchers: HashMap<ModelKey, DispatcherSnapshot>,
    pub unresolved_failures: [u64; 5],
    pub cache: CacheStats,
    pub resources: ResourceSnapshot,
    pub loaded_models: usize,
}

/// Renders a report as Prometheus text exposition.
pub fn render_prometheus(report: &PrometheusReport) -> String {
    let mut out = String::with_capacity(4_096);

    metric(
        &mut out,
        "gm_server_info",
        "gauge",
        "Server build information; the value is always 1.",
    );
    let _ = writeln!(
        out,
        "gm_server_info{{version=\"{}\",platform=\"{}\",os=\"{}\"}} 1",
        report.server_version, report.runtime.platform, report.runtime.os
    );

    // The version and the platform are also published one label each. A join
    // against gm_server_info works but is awkward in an alert and impossible
    // in a Grafana variable, and these two are what a deployment is grouped
    // by: which build is running, and on what.
    metric(
        &mut out,
        "gm_server_version",
        "gauge",
        "Server version; the value is always 1 and the version is the label.",
    );
    let _ = writeln!(
        out,
        "gm_server_version{{version=\"{}\"}} 1",
        report.server_version
    );

    metric(
        &mut out,
        "gm_server_platform",
        "gauge",
        "Architecture and operating system this server is running on, named as \
         container images name them (amd64, arm64); the value is always 1.",
    );
    let _ = writeln!(
        out,
        "gm_server_platform{{platform=\"{}\",os=\"{}\"}} 1",
        report.runtime.platform, report.runtime.os
    );

    metric(
        &mut out,
        "gm_server_allocator",
        "gauge",
        "Memory allocator this build uses: `system` for the platform's own, \
         `jemalloc` for a build that returns unused pages on a decay timer. \
         It decides whether resident memory falls when the server goes idle, \
         so a memory graph cannot be read without it. The value is always 1.",
    );
    let _ = writeln!(
        out,
        "gm_server_allocator{{name=\"{}\"}} 1",
        report.runtime.allocator
    );

    // -- how the machine is being used ------------------------------------
    //
    // Throughput that stops climbing with concurrency is either a saturated
    // machine or a server that was told to use one core; these say which.
    // gm_cpu_count is reported with the other resource readings below.
    for (name, help, value) in [
        (
            "gm_server_worker_threads",
            "Async worker threads serving requests.",
            report.runtime.worker_threads as u64,
        ),
        (
            "gm_server_blocking_threads",
            "Threads available for blocking work. A batch executes on one of \
             these, so this bounds how many inferences can run at once.",
            report.runtime.blocking_threads as u64,
        ),
        (
            "gm_server_intra_op_threads",
            "Threads ONNX Runtime may use inside one operator, per model. \
             0 means ONNX Runtime chooses, which is one per core per model.",
            report.runtime.intra_op_threads as u64,
        ),
        (
            "gm_server_compute_threads_max",
            "Upper bound on threads that may run model code at once: \
             blocking threads times intra-op threads. Above gm_cpu_count the \
             machine is oversubscribed.",
            report.runtime.compute_threads() as u64,
        ),
        (
            "gm_malloc_arena_max",
            "The MALLOC_ARENA_MAX the process was started with, or 0 when \
             unset. Unset lets glibc keep up to eight freed-memory arenas per \
             core, which is the largest single influence on resident memory \
             under load.",
            report.runtime.malloc_arena_max.unwrap_or(0) as u64,
        ),
        (
            "gm_server_cpu_mem_arena",
            "1 when ONNX Runtime's CPU arena allocator is in use. The arena \
             never returns memory, so resident memory grows to the high-water \
             mark of the largest batch seen.",
            u64::from(report.runtime.cpu_mem_arena),
        ),
    ] {
        metric(&mut out, name, "gauge", help);
        let _ = writeln!(out, "{name} {value}");
    }

    // -- per-model request counts -----------------------------------------

    metric(
        &mut out,
        "gm_inference_request_success",
        "counter",
        "Successful inference requests, counting a batched request as one.",
    );
    for (key, model) in &report.models {
        sample(&mut out, "gm_inference_request_success", key, model.success);
    }

    metric(
        &mut out,
        "gm_inference_request_failure",
        "counter",
        "Failed inference requests, by reason.",
    );
    for (key, model) in &report.models {
        for reason in FailureReason::ALL {
            let _ = writeln!(
                out,
                "gm_inference_request_failure{{model=\"{}\",version=\"{}\",reason=\"{}\"}} {}",
                key.name,
                key.version,
                reason.label(),
                model.failures(reason)
            );
        }
    }
    for reason in FailureReason::ALL {
        let _ = writeln!(
            out,
            "gm_inference_request_failure{{model=\"\",version=\"\",reason=\"{}\"}} {}",
            reason.label(),
            report.unresolved_failures[reason.index()]
        );
    }

    metric(
        &mut out,
        "gm_inference_count",
        "counter",
        "Inferences performed, counting a batch of n rows as n.",
    );
    for (key, dispatcher) in sorted(&report.dispatchers) {
        sample(&mut out, "gm_inference_count", key, dispatcher.rows);
    }

    metric(
        &mut out,
        "gm_inference_exec_count",
        "counter",
        "Batch executions. Inference count divided by this is the mean batch size.",
    );
    for (key, dispatcher) in sorted(&report.dispatchers) {
        sample(&mut out, "gm_inference_exec_count", key, dispatcher.batches);
    }

    metric(
        &mut out,
        "gm_inference_pending_request_count",
        "gauge",
        "Requests queued but not yet being executed.",
    );
    for (key, dispatcher) in sorted(&report.dispatchers) {
        sample(
            &mut out,
            "gm_inference_pending_request_count",
            key,
            dispatcher.pending,
        );
    }

    // -- per-model latencies ----------------------------------------------

    metric(
        &mut out,
        "gm_inference_request_duration_us",
        "counter",
        "Cumulative end-to-end request handling time in microseconds.",
    );
    for (key, model) in &report.models {
        sample(
            &mut out,
            "gm_inference_request_duration_us",
            key,
            model.request_duration_micros,
        );
    }

    metric(
        &mut out,
        "gm_inference_queue_duration_us",
        "counter",
        "Cumulative time requests spent waiting in the batching queue, in microseconds.",
    );
    for (key, dispatcher) in sorted(&report.dispatchers) {
        sample(
            &mut out,
            "gm_inference_queue_duration_us",
            key,
            dispatcher.queue_duration_micros,
        );
    }

    metric(
        &mut out,
        "gm_inference_compute_infer_duration_us",
        "counter",
        "Cumulative time spent executing batches in the backend, in microseconds.",
    );
    for (key, dispatcher) in sorted(&report.dispatchers) {
        sample(
            &mut out,
            "gm_inference_compute_infer_duration_us",
            key,
            dispatcher.compute_duration_micros,
        );
    }

    metric(
        &mut out,
        "gm_inference_request_duration_seconds",
        "histogram",
        "Distribution of end-to-end request handling time.",
    );
    for (key, model) in &report.models {
        let mut cumulative = 0u64;
        for (index, bound) in DURATION_BUCKETS_SECS.iter().enumerate() {
            cumulative += model.histogram[index];
            let _ = writeln!(
                out,
                "gm_inference_request_duration_seconds_bucket\
                 {{model=\"{}\",version=\"{}\",le=\"{bound}\"}} {cumulative}",
                key.name, key.version
            );
        }
        let _ = writeln!(
            out,
            "gm_inference_request_duration_seconds_bucket\
             {{model=\"{}\",version=\"{}\",le=\"+Inf\"}} {}",
            key.name, key.version, model.histogram_count
        );
        let _ = writeln!(
            out,
            "gm_inference_request_duration_seconds_sum{{model=\"{}\",version=\"{}\"}} {}",
            key.name,
            key.version,
            model.histogram_sum_micros as f64 / 1e6
        );
        let _ = writeln!(
            out,
            "gm_inference_request_duration_seconds_count{{model=\"{}\",version=\"{}\"}} {}",
            key.name, key.version, model.histogram_count
        );
    }

    // -- this server's own batching ---------------------------------------

    metric(
        &mut out,
        "gm_batch_merged_count",
        "counter",
        "Batch executions that combined more than one request.",
    );
    for (key, dispatcher) in sorted(&report.dispatchers) {
        sample(
            &mut out,
            "gm_batch_merged_count",
            key,
            dispatcher.merged_batches,
        );
    }

    // -- the model cache ---------------------------------------------------

    let cache = &report.cache;
    for (name, help, value) in [
        (
            "gm_model_cache_entries",
            "Models currently loaded in memory.",
            cache.entries as u64,
        ),
        (
            "gm_model_cache_resident_bytes",
            "Bytes of model artifacts currently loaded.",
            cache.resident_bytes as u64,
        ),
        (
            "gm_models_available",
            "Model versions the repository holds.",
            report.loaded_models as u64,
        ),
    ] {
        metric(&mut out, name, "gauge", help);
        let _ = writeln!(out, "{name} {value}");
    }
    for (name, help, value) in [
        (
            "gm_model_cache_hit_count",
            "Requests served by an already loaded model.",
            cache.hits,
        ),
        (
            "gm_model_cache_miss_count",
            "Requests that had to load a model.",
            cache.misses,
        ),
        (
            "gm_model_cache_eviction_count",
            "Models unloaded to make room for another.",
            cache.evictions,
        ),
        (
            "gm_model_cache_expiration_count",
            "Models unloaded after sitting unused past the time-to-live.",
            cache.expirations,
        ),
    ] {
        metric(&mut out, name, "counter", help);
        let _ = writeln!(out, "{name} {value}");
    }

    // -- process resources -------------------------------------------------

    let resources = &report.resources;
    metric(
        &mut out,
        "gm_cpu_utilization",
        "gauge",
        "Share of the machine's CPU this process used over the last interval, \
         where 1 is every core fully busy.",
    );
    let _ = writeln!(out, "gm_cpu_utilization {:.6}", resources.cpu_utilization);

    metric(&mut out, "gm_cpu_count", "gauge", "Cores available.");
    let _ = writeln!(out, "gm_cpu_count {}", resources.cpu_count);

    metric(
        &mut out,
        "gm_memory_usage_bytes",
        "gauge",
        "Resident memory of this process.",
    );
    let _ = writeln!(out, "gm_memory_usage_bytes {}", resources.memory_bytes);

    metric(
        &mut out,
        "gm_memory_peak_bytes",
        "gauge",
        "Largest resident memory this process has held. Compared with \
         gm_memory_usage_bytes it separates memory that was given back from \
         memory that was not.",
    );
    let _ = writeln!(out, "gm_memory_peak_bytes {}", resources.memory_peak_bytes);

    metric(
        &mut out,
        "gm_memory_total_bytes",
        "gauge",
        "Memory the machine has in total.",
    );
    let _ = writeln!(
        out,
        "gm_memory_total_bytes {}",
        resources.memory_total_bytes
    );

    metric(
        &mut out,
        "gm_memory_utilization",
        "gauge",
        "Resident memory as a share of the machine's total.",
    );
    let _ = writeln!(
        out,
        "gm_memory_utilization {:.6}",
        resources.memory_utilization
    );

    metric(
        &mut out,
        "gm_server_uptime_seconds",
        "gauge",
        "Seconds since the server started.",
    );
    let _ = writeln!(out, "gm_server_uptime_seconds {:.3}", resources.uptime_secs);

    out
}

fn metric(out: &mut String, name: &str, kind: &str, help: &str) {
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} {kind}");
}

fn sample(out: &mut String, name: &str, key: &ModelKey, value: u64) {
    let _ = writeln!(
        out,
        "{name}{{model=\"{}\",version=\"{}\"}} {value}",
        key.name, key.version
    );
}

/// Dispatcher snapshots in model order, so a scrape is stable between calls.
fn sorted(
    dispatchers: &HashMap<ModelKey, DispatcherSnapshot>,
) -> Vec<(&ModelKey, &DispatcherSnapshot)> {
    let mut entries: Vec<_> = dispatchers.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    entries
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(name: &str) -> ModelKey {
        ModelKey::new(name, "1")
    }

    fn report(
        models: Vec<(ModelKey, ModelMetricsSnapshot)>,
        dispatchers: Vec<(ModelKey, DispatcherSnapshot)>,
    ) -> PrometheusReport {
        PrometheusReport {
            server_version: "0.2.0",
            runtime: RuntimeInfo {
                version: "0.2.0",
                platform: "amd64",
                os: "linux",
                cpu_count: 4,
                worker_threads: 4,
                blocking_threads: 8,
                intra_op_threads: 1,
                cpu_mem_arena: false,
                malloc_arena_max: Some(2),
                allocator: crate::runtime::SYSTEM_ALLOCATOR,
            },
            models,
            dispatchers: dispatchers.into_iter().collect(),
            unresolved_failures: [0; 5],
            cache: CacheStats {
                entries: 2,
                hits: 10,
                misses: 2,
                evictions: 1,
                expirations: 0,
                resident_bytes: 4_096,
            },
            resources: ResourceSnapshot {
                cpu_utilization: 0.25,
                memory_bytes: 1_048_576,
                memory_peak_bytes: 2_097_152,
                memory_total_bytes: 8_388_608,
                memory_utilization: 0.125,
                cpu_count: 4,
                uptime_secs: 12.5,
            },
            loaded_models: 2,
        }
    }

    #[test]
    fn a_failure_is_classified_by_who_has_to_fix_it() {
        assert_eq!(
            FailureReason::of(&InferError::Overloaded),
            FailureReason::Rejected
        );
        assert_eq!(
            FailureReason::of(&InferError::Timeout(Duration::from_secs(1))),
            FailureReason::Timeout
        );
        assert_eq!(
            FailureReason::of(&InferError::Backend("boom".into())),
            FailureReason::Backend
        );
        assert_eq!(
            FailureReason::of(&InferError::ModelNotFound("m".into())),
            FailureReason::Invalid
        );
        assert_eq!(
            FailureReason::of(&InferError::remote_server_error("boom")),
            FailureReason::Backend
        );
        assert_eq!(
            FailureReason::of(&InferError::Internal("x".into())),
            FailureReason::Other
        );
    }

    #[tokio::test]
    async fn successes_and_failures_are_counted_per_model() {
        let registry = MetricsRegistry::new();
        registry
            .record_success(&key("energy"), Duration::from_micros(500))
            .await;
        registry
            .record_success(&key("energy"), Duration::from_micros(1_500))
            .await;
        registry
            .record_failure(
                Some(&key("energy")),
                &InferError::Overloaded,
                Duration::ZERO,
            )
            .await;
        registry
            .record_failure(
                Some(&key("bands")),
                &InferError::Backend("x".into()),
                Duration::ZERO,
            )
            .await;

        let snapshot = registry.snapshot().await;
        assert_eq!(snapshot.len(), 2);
        assert_eq!(snapshot[0].0, key("bands"));

        let energy = snapshot[1].1;
        assert_eq!(energy.success, 2);
        assert_eq!(energy.failures(FailureReason::Rejected), 1);
        assert_eq!(energy.total_failures(), 1);
        assert_eq!(energy.request_duration_micros, 2_000);
        assert_eq!(energy.histogram_count, 3);
    }

    #[tokio::test]
    async fn a_request_that_never_reached_a_model_is_counted_separately() {
        let registry = MetricsRegistry::new();
        registry
            .record_failure(
                None,
                &InferError::ModelNotFound("ghost".into()),
                Duration::ZERO,
            )
            .await;

        assert!(registry.snapshot().await.is_empty());
        assert_eq!(
            registry.unresolved_failures()[FailureReason::Invalid.index()],
            1
        );
    }

    #[tokio::test]
    async fn observations_land_in_the_bucket_that_bounds_them() {
        let registry = MetricsRegistry::new();
        for micros in [50, 300, 3_000, 2_000_000] {
            registry
                .record_success(&key("energy"), Duration::from_micros(micros))
                .await;
        }

        let snapshot = registry.snapshot().await[0].1;
        assert_eq!(snapshot.histogram_count, 4);
        assert_eq!(snapshot.histogram_sum_micros, 2_003_350);
        // The 2-second observation exceeds every bound, so it appears only in
        // the count, which is what +Inf reports.
        assert_eq!(snapshot.histogram.iter().sum::<u64>(), 3);
    }

    #[test]
    fn the_exposition_names_follow_tritons_with_a_gm_prefix() {
        let text = render_prometheus(&report(
            vec![(key("energy"), ModelMetricsSnapshot::default())],
            vec![(key("energy"), DispatcherSnapshot::default())],
        ));

        for name in [
            "gm_inference_request_success",
            "gm_inference_request_failure",
            "gm_inference_count",
            "gm_inference_exec_count",
            "gm_inference_pending_request_count",
            "gm_inference_request_duration_us",
            "gm_inference_queue_duration_us",
            "gm_inference_compute_infer_duration_us",
        ] {
            assert!(text.contains(&format!("# TYPE {name} ")), "missing {name}");
        }
        assert!(!text.contains("nv_"), "Triton's prefix must not appear");
    }

    #[test]
    fn metrics_that_cannot_apply_to_this_server_are_absent() {
        let text = render_prometheus(&report(Vec::new(), Vec::new()));
        for absent in ["gpu", "pinned_memory", "response_cache"] {
            assert!(!text.contains(absent), "{absent} has no meaning here");
        }
    }

    #[test]
    fn every_series_is_labelled_by_model_and_version() {
        let mut model = ModelMetricsSnapshot::default();
        model.success = 7;
        let dispatcher = DispatcherSnapshot {
            rows: 20,
            batches: 5,
            pending: 3,
            queue_duration_micros: 400,
            compute_duration_micros: 900,
            merged_batches: 4,
            ..Default::default()
        };
        let text = render_prometheus(&report(
            vec![(key("energy"), model)],
            vec![(key("energy"), dispatcher)],
        ));

        assert!(text.contains("gm_inference_request_success{model=\"energy\",version=\"1\"} 7"));
        assert!(text.contains("gm_inference_count{model=\"energy\",version=\"1\"} 20"));
        assert!(text.contains("gm_inference_exec_count{model=\"energy\",version=\"1\"} 5"));
        assert!(
            text.contains("gm_inference_pending_request_count{model=\"energy\",version=\"1\"} 3")
        );
        assert!(text.contains("gm_inference_queue_duration_us{model=\"energy\",version=\"1\"} 400"));
        assert!(text.contains(
            "gm_inference_compute_infer_duration_us{model=\"energy\",version=\"1\"} 900"
        ));
        assert!(text.contains("gm_batch_merged_count{model=\"energy\",version=\"1\"} 4"));
    }

    #[test]
    fn every_failure_reason_is_reported_even_at_zero() {
        let text = render_prometheus(&report(
            vec![(key("energy"), ModelMetricsSnapshot::default())],
            Vec::new(),
        ));
        for reason in FailureReason::ALL {
            assert!(
                text.contains(&format!(
                    "gm_inference_request_failure{{model=\"energy\",version=\"1\",reason=\"{}\"}} 0",
                    reason.label()
                )),
                "missing reason {}",
                reason.label()
            );
        }
    }

    #[test]
    fn histogram_buckets_are_cumulative_and_end_at_the_total() {
        let mut model = ModelMetricsSnapshot::default();
        model.histogram[0] = 2;
        model.histogram[3] = 1;
        model.histogram_count = 4; // one observation exceeded every bound
        model.histogram_sum_micros = 2_500_000;

        let text = render_prometheus(&report(vec![(key("energy"), model)], Vec::new()));
        assert!(text.contains("le=\"0.0001\"} 2"), "{text}");
        assert!(text.contains("le=\"0.001\"} 3"), "{text}");
        assert!(text.contains("le=\"+Inf\"} 4"), "{text}");
        assert!(
            text.contains("_sum{model=\"energy\",version=\"1\"} 2.5"),
            "{text}"
        );
        assert!(
            text.contains("_count{model=\"energy\",version=\"1\"} 4"),
            "{text}"
        );
    }

    #[test]
    fn cache_and_resource_readings_are_reported() {
        let text = render_prometheus(&report(Vec::new(), Vec::new()));
        assert!(text.contains("gm_model_cache_entries 2"));
        assert!(text.contains("gm_model_cache_hit_count 10"));
        assert!(text.contains("gm_model_cache_miss_count 2"));
        assert!(text.contains("gm_model_cache_eviction_count 1"));
        assert!(text.contains("gm_model_cache_resident_bytes 4096"));
        assert!(text.contains("gm_cpu_utilization 0.250000"));
        assert!(text.contains("gm_cpu_count 4"));
        assert!(text.contains("gm_memory_usage_bytes 1048576"));
        assert!(text.contains("gm_memory_peak_bytes 2097152"));
        assert!(text.contains("gm_memory_utilization 0.125000"));
        assert!(
            text.contains("gm_server_info{version=\"0.2.0\",platform=\"amd64\",os=\"linux\"} 1")
        );
    }

    #[test]
    fn every_metric_is_declared_before_its_samples() {
        let text = render_prometheus(&report(
            vec![(key("energy"), ModelMetricsSnapshot::default())],
            vec![(key("energy"), DispatcherSnapshot::default())],
        ));

        let mut declared = Vec::new();
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("# TYPE ") {
                declared.push(rest.split(' ').next().unwrap().to_string());
                continue;
            }
            if line.starts_with('#') || line.is_empty() {
                continue;
            }
            let name = line.split(['{', ' ']).next().unwrap();

            // A histogram declares one name but emits three suffixed series.
            let base = ["_bucket", "_sum", "_count"]
                .iter()
                .find_map(|suffix| name.strip_suffix(suffix))
                .unwrap_or(name);
            assert!(
                declared.iter().any(|d| d == name || d == base),
                "sample for undeclared metric: {line}"
            );
        }
    }
    /// A deployment is grouped by which build is running and on what, so both
    /// are published as series of their own rather than only as labels on
    /// gm_server_info.
    #[test]
    fn the_build_and_the_platform_are_each_a_series_of_their_own() {
        let text = render_prometheus(&report(Vec::new(), Vec::new()));

        assert!(
            text.contains("gm_server_version{version=\"0.2.0\"} 1"),
            "{text}"
        );
        assert!(
            text.contains("gm_server_platform{platform=\"amd64\",os=\"linux\"} 1"),
            "{text}"
        );
        assert!(
            text.contains("gm_server_info{version=\"0.2.0\",platform=\"amd64\",os=\"linux\"} 1"),
            "{text}"
        );
    }

    /// Whether the machine is oversubscribed is a question about threads, so
    /// the thread plan is reported rather than left to the startup log.
    #[test]
    fn the_thread_plan_is_reported_so_oversubscription_can_be_seen() {
        let mut report = report(Vec::new(), Vec::new());
        report.runtime.worker_threads = 8;
        report.runtime.blocking_threads = 8;
        report.runtime.intra_op_threads = 2;
        let text = render_prometheus(&report);

        assert!(text.contains("\ngm_cpu_count 4\n"), "{text}");
        assert!(text.contains("\ngm_server_worker_threads 8\n"), "{text}");
        assert!(text.contains("\ngm_server_blocking_threads 8\n"), "{text}");
        assert!(text.contains("\ngm_server_intra_op_threads 2\n"), "{text}");
        // 8 blocking threads that may each use 2 operator threads on a
        // 4-core machine.
        assert!(
            text.contains("\ngm_server_compute_threads_max 16\n"),
            "{text}"
        );
        assert!(text.contains("\ngm_server_cpu_mem_arena 0\n"), "{text}");
        assert!(text.contains("\ngm_malloc_arena_max 2\n"), "{text}");
        // Which allocator is running decides whether an idle server gives
        // memory back, so it is reported rather than left to be guessed.
        assert!(
            text.contains("gm_server_allocator{name=\"system\"} 1")
                || text.contains("gm_server_allocator{name=\"jemalloc\"} 1"),
            "{text}"
        );
    }
}
