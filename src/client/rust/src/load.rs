//! Driving load at a server and recording what happened.
//!
//! The report is written as JSON so a run can be loaded into a timeseries
//! store: a summary with latency percentiles, and one tick per metrics poll
//! carrying both what the client measured and what the server reported about
//! itself at that moment. Having both sides on one timeline is what makes a
//! result interpretable — client latency alone cannot tell a slow model from a
//! saturated queue.

use crate::rows::synthesize_inputs;
use crate::{scrape, PredictV2Client, Scrape};
use galemind_core::{InferError, InferRequest, ModelRef, Parameters};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};

type Result<T> = std::result::Result<T, InferError>;

/// How to drive the load.
#[derive(Debug, Clone)]
pub struct LoadConfig {
    pub model: String,
    pub version: Option<String>,
    /// Requests in flight at once.
    pub concurrency: usize,
    /// Stop after this many requests, if given.
    pub total_requests: Option<usize>,
    /// Stop after this long, if given. One of the two must be set.
    pub duration: Option<Duration>,
    /// Rows per request, so client-side batching can be varied too.
    pub rows_per_request: usize,
    /// Requests to run and discard before measuring, so that model loading and
    /// connection setup do not land in the percentiles.
    pub warmup_requests: usize,
    pub seed: u64,
    /// How often to poll the server's metrics. `None` records no timeseries.
    pub metrics_interval: Option<Duration>,
    /// Where to poll metrics from, which is the HTTP endpoint even when the
    /// load itself runs over gRPC.
    pub metrics_endpoint: Option<String>,
    /// Keep every latency in the report, not just the percentiles.
    ///
    /// Percentiles cannot be averaged -- the median of two medians is not a
    /// median -- so merging the reports of several processes into one needs
    /// the samples themselves. Off by default, because a long run's samples
    /// are a megabyte of JSON nobody reads.
    pub record_samples: bool,
}

impl Default for LoadConfig {
    fn default() -> Self {
        Self {
            model: "building_energy".to_string(),
            version: None,
            concurrency: 16,
            total_requests: Some(1_000),
            duration: None,
            rows_per_request: 1,
            warmup_requests: 32,
            seed: 1,
            metrics_interval: Some(Duration::from_millis(500)),
            metrics_endpoint: None,
            record_samples: false,
        }
    }
}

/// Latency percentiles, in milliseconds.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Percentiles {
    pub min_ms: f64,
    pub p50_ms: f64,
    pub p55_ms: f64,
    pub p90_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
    pub mean_ms: f64,
}

impl Percentiles {
    /// Computes percentiles from latencies in milliseconds.
    ///
    /// The nearest-rank method is used: the value at the smallest index whose
    /// share of the sorted samples reaches the requested rank. It needs no
    /// interpolation, so every reported figure is a latency that really
    /// happened.
    pub fn of(mut latencies_ms: Vec<f64>) -> Self {
        if latencies_ms.is_empty() {
            return Self::default();
        }
        latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let at = |quantile: f64| -> f64 {
            // The epsilon matters: 0.55 * 100 is 55.000000000000014 in binary
            // floating point, and rounding that up would report the 56th
            // sample as the 55th percentile.
            let rank = (quantile * latencies_ms.len() as f64 - 1e-9)
                .ceil()
                .max(1.0) as usize;
            let index = (rank - 1).min(latencies_ms.len() - 1);
            latencies_ms[index]
        };

        Self {
            min_ms: latencies_ms[0],
            p50_ms: at(0.50),
            p55_ms: at(0.55),
            p90_ms: at(0.90),
            p95_ms: at(0.95),
            p99_ms: at(0.99),
            max_ms: latencies_ms[latencies_ms.len() - 1],
            mean_ms: latencies_ms.iter().sum::<f64>() / latencies_ms.len() as f64,
        }
    }
}

/// What a run achieved.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Summary {
    pub completed: u64,
    pub failed: u64,
    pub wall_secs: f64,
    /// Successful requests per second.
    pub throughput_rps: f64,
    /// Rows inferred per second, which is the figure batching improves.
    pub rows_per_sec: f64,
    pub latency: Percentiles,
    /// Failures by kind, so a run that fell over says how.
    pub errors: BTreeMap<String, u64>,
}

/// One point on the timeline, pairing client and server views.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Tick {
    /// Milliseconds since the measured phase began.
    pub elapsed_ms: u64,
    /// Wall-clock time, so a run can be aligned with other systems' data.
    pub unix_ms: u64,

    // What the client can see.
    pub completed: u64,
    pub failed: u64,
    pub in_flight: u64,
    /// Requests completed since the previous tick, divided by the gap.
    pub throughput_rps: f64,

    // What the server reported about itself.
    pub cpu_utilization: Option<f64>,
    pub memory_bytes: Option<f64>,
    pub memory_utilization: Option<f64>,
    pub server_inference_count: Option<f64>,
    pub server_exec_count: Option<f64>,
    pub server_pending: Option<f64>,
    pub server_queue_duration_us: Option<f64>,
    pub server_compute_duration_us: Option<f64>,
    /// Inference count over execution count: the mean batch size the server
    /// actually formed, which is how adaptive batching is observed working.
    pub mean_batch_size: Option<f64>,
}

/// A whole run, ready to be written as JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadReport {
    pub model: String,
    pub transport: String,
    pub concurrency: usize,
    pub rows_per_request: usize,
    pub started_unix_ms: u64,
    pub summary: Summary,
    pub timeseries: Vec<Tick>,
    /// Processes this report covers: one, unless it is a merged report.
    #[serde(default = "one")]
    pub processes: usize,
    /// Sockets the run held open, where the client could say.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sockets: Option<usize>,
    /// Every latency measured, in milliseconds, when the run was asked to
    /// keep them. Present so that reports can be merged exactly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub samples_ms: Option<Vec<f64>>,
}

fn one() -> usize {
    1
}

impl LoadReport {
    /// The report as pretty JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
    }

    /// Merges the reports of several processes that ran the same load at the
    /// same time into one.
    ///
    /// The sums are sums: completed, failed, sockets and the error tallies
    /// all add up, and so does the concurrency, because that is what running
    /// ten processes at a thousand each was for.
    ///
    /// The rates are computed against the longest wall time rather than
    /// added, since the processes overlapped: adding their throughputs would
    /// be right only if they had run one after another, and dividing by the
    /// longest is the conservative reading of what the server sustained while
    /// all of them were on it.
    ///
    /// The percentiles are recomputed from the pooled samples where the
    /// processes kept them. Where they did not, the field is left at the
    /// first report's, and the caller is told by
    /// [`Summary::latency`] being that of one process -- which is why
    /// `--processes` asks for samples.
    pub fn merge(reports: Vec<LoadReport>) -> Option<LoadReport> {
        let mut reports = reports.into_iter();
        let first = reports.next()?;
        let mut merged = first.clone();
        let mut longest = first.summary.wall_secs;
        let mut samples = first.samples_ms.clone().unwrap_or_default();
        let mut rows = first.summary.rows_per_sec * first.summary.wall_secs;

        merged.processes = 1;
        for report in reports {
            merged.processes += 1;
            merged.concurrency += report.concurrency;
            merged.summary.completed += report.summary.completed;
            merged.summary.failed += report.summary.failed;
            merged.sockets = match (merged.sockets, report.sockets) {
                (Some(a), Some(b)) => Some(a + b),
                (a, b) => a.or(b),
            };
            for (kind, count) in report.summary.errors {
                *merged.summary.errors.entry(kind).or_default() += count;
            }
            rows += report.summary.rows_per_sec * report.summary.wall_secs;
            longest = longest.max(report.summary.wall_secs);
            if let Some(mut theirs) = report.samples_ms {
                samples.append(&mut theirs);
            }
            // The timeseries of the one process that polled, if any did.
            if merged.timeseries.is_empty() {
                merged.timeseries = report.timeseries;
            }
            merged.started_unix_ms = merged.started_unix_ms.min(report.started_unix_ms);
        }

        merged.summary.wall_secs = longest;
        merged.summary.throughput_rps = if longest > 0.0 {
            merged.summary.completed as f64 / longest
        } else {
            0.0
        };
        merged.summary.rows_per_sec = if longest > 0.0 { rows / longest } else { 0.0 };
        if !samples.is_empty() {
            merged.summary.latency = Percentiles::of(samples.clone());
            merged.samples_ms = merged.samples_ms.as_ref().map(|_| samples);
        }
        Some(merged)
    }
}

#[derive(Default)]
struct Counters {
    /// Slots claimed by workers. Claiming before sending is what keeps a
    /// bounded run exact rather than rounding up to a multiple of the
    /// concurrency.
    issued: AtomicU64,
    completed: AtomicU64,
    failed: AtomicU64,
    in_flight: AtomicU64,
    rows: AtomicU64,
    stop: AtomicUsize,
}

impl Counters {
    fn should_stop(&self) -> bool {
        self.stop.load(Ordering::Relaxed) != 0
    }
}

fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or_default()
}

/// Shortens a failure to a label suitable for grouping.
fn error_label(error: &InferError) -> String {
    match error {
        InferError::Overloaded => "overloaded".to_string(),
        InferError::Timeout(_) => "timeout".to_string(),
        InferError::ModelNotFound(_) => "model_not_found".to_string(),
        InferError::VersionNotFound { .. } => "version_not_found".to_string(),
        InferError::InvalidRequest(_) => "invalid_request".to_string(),
        InferError::ModelNotReady(_) => "model_not_ready".to_string(),
        InferError::Unsupported(_) => "unsupported".to_string(),
        InferError::Backend(_) => "backend".to_string(),
        InferError::Store(_) => "store".to_string(),
        InferError::Internal(_) => "internal".to_string(),
        InferError::Remote { client_error, .. } => {
            if *client_error {
                "remote_client_error".to_string()
            } else {
                "remote_server_error".to_string()
            }
        }
    }
}

/// Runs the load and returns the report.
pub async fn run_load(client: Arc<dyn PredictV2Client>, config: LoadConfig) -> Result<LoadReport> {
    if config.total_requests.is_none() && config.duration.is_none() {
        return Err(InferError::InvalidRequest(
            "a load run needs either a request count or a duration".to_string(),
        ));
    }
    let concurrency = config.concurrency.max(1);

    // Checked before anything is sent. A run that cannot fit in the process'
    // open-file limit fails somewhere inside a connection pool, with an error
    // that reads like a bug in the client; this refuses it up front and says
    // which limit and how to raise it.
    // Checked against what the client will actually open, which is the
    // concurrency only for a transport that gives a request its own socket.
    let sockets = client.sockets_for(concurrency);
    if let Some(warning) = crate::limits::check_sockets(concurrency, sockets)? {
        warn!("{warning}");
    }

    // The request is built once from the model's own metadata, so every worker
    // sends something the model actually accepts -- every input of it, not
    // just the first, so that a generator taking an embedding and a noise
    // vector can be driven like anything else.
    let model = ModelRef::new(config.model.clone(), config.version.clone());
    let metadata = client.model_metadata(&model).await?;
    let template = InferRequest {
        model: model.clone(),
        id: None,
        parameters: Parameters::new(),
        inputs: synthesize_inputs(&metadata, config.rows_per_request.max(1), config.seed)?,
        requested_outputs: Vec::new(),
    };

    // Warm up, so model loading and connection setup stay out of the
    // percentiles. Failures here are reported, not swallowed: a run whose
    // warmup cannot complete has nothing to measure.
    //
    // The warmup runs at the concurrency the run will use, which is the point
    // of it for anything above a handful of workers. Sent one at a time it
    // establishes exactly one connection, and the measured phase then pays
    // for the other 699 handshakes inside its own latencies -- the opposite
    // of what a warmup is for. The count is raised to cover the concurrency
    // for the same reason: a warmup shorter than the concurrency leaves some
    // workers to connect while being measured.
    let warmup_requests = match config.warmup_requests {
        // Explicitly asked for no warmup, which is what a run that wants the
        // cost of a cold model in its percentiles wants.
        0 => 0,
        asked => asked.max(concurrency),
    };
    if warmup_requests > 0 {
        // Where the transport multiplexes, the sockets are established
        // first, at the socket count rather than at the concurrency.
        //
        // This is what keeps a multiplexed run on the few sockets it is
        // supposed to use. A pool with nothing in it, handed ten thousand
        // requests at once, has no connection to put them on and dials one
        // for each until some are ready -- a connection storm that settles
        // back down afterwards but leaves thousands of sockets open at its
        // peak, which is the cost multiplexing was chosen to avoid. Filling
        // the pool first means the burst arrives to find connections ready
        // and becomes streams on them.
        if sockets < concurrency {
            warm_up(&client, &template, sockets, sockets).await?;
            debug!(sockets, "primed the connection pool");
        }
        warm_up(&client, &template, warmup_requests, concurrency).await?;
    }
    info!(
        warmup = warmup_requests,
        concurrency,
        sockets,
        transport = client.transport(),
        "warmed up, starting measured phase"
    );

    let counters = Arc::new(Counters::default());
    let latencies = Arc::new(Mutex::new(Vec::<f64>::with_capacity(
        config.total_requests.unwrap_or(10_000),
    )));
    let errors = Arc::new(Mutex::new(BTreeMap::<String, u64>::new()));
    let started = Instant::now();
    let started_unix_ms = unix_ms();

    // Poll the server's own metrics alongside the load, so the two views share
    // one timeline.
    let ticks = Arc::new(Mutex::new(Vec::<Tick>::new()));
    let poller = config.metrics_interval.and_then(|interval| {
        let endpoint = config.metrics_endpoint.clone()?;
        Some(tokio::spawn(poll_metrics(
            endpoint,
            interval,
            started,
            Arc::clone(&counters),
            Arc::clone(&ticks),
            config.model.clone(),
        )))
    });

    // A deadline, when one was asked for.
    if let Some(duration) = config.duration {
        let counters = Arc::clone(&counters);
        tokio::spawn(async move {
            tokio::time::sleep(duration).await;
            counters.stop.store(1, Ordering::Relaxed);
        });
    }

    let total_requests = config.total_requests.map(|total| total as u64);
    let rows_per_request = config.rows_per_request.max(1) as u64;

    let workers: Vec<_> = (0..concurrency)
        .map(|_| {
            let client = Arc::clone(&client);
            let counters = Arc::clone(&counters);
            let latencies = Arc::clone(&latencies);
            let errors = Arc::clone(&errors);
            let request = clone_request(&template);

            tokio::spawn(async move {
                loop {
                    if counters.should_stop() {
                        return;
                    }
                    // Claim this request before sending it. Whoever claims the
                    // last slot sends it; everyone else goes home.
                    if let Some(total) = total_requests {
                        if counters.issued.fetch_add(1, Ordering::Relaxed) >= total {
                            return;
                        }
                    }

                    counters.in_flight.fetch_add(1, Ordering::Relaxed);
                    let at = Instant::now();
                    let outcome = client.infer(clone_request(&request)).await;
                    let elapsed = at.elapsed().as_secs_f64() * 1_000.0;
                    counters.in_flight.fetch_sub(1, Ordering::Relaxed);

                    match outcome {
                        Ok(_) => {
                            counters.completed.fetch_add(1, Ordering::Relaxed);
                            counters.rows.fetch_add(rows_per_request, Ordering::Relaxed);
                            if let Ok(mut latencies) = latencies.lock() {
                                latencies.push(elapsed);
                            }
                        }
                        Err(e) => {
                            counters.failed.fetch_add(1, Ordering::Relaxed);
                            if let Ok(mut errors) = errors.lock() {
                                *errors.entry(error_label(&e)).or_insert(0) += 1;
                            }
                        }
                    }
                }
            })
        })
        .collect();

    for worker in workers {
        let _ = worker.await;
    }
    counters.stop.store(1, Ordering::Relaxed);
    let wall_secs = started.elapsed().as_secs_f64();

    if let Some(poller) = poller {
        // One last tick, so the report covers the end of the run.
        let _ = tokio::time::timeout(Duration::from_secs(2), poller).await;
    }

    let completed = counters.completed.load(Ordering::Relaxed);
    let failed = counters.failed.load(Ordering::Relaxed);
    let total_rows = counters.rows.load(Ordering::Relaxed);
    let latencies = latencies.lock().map(|l| l.clone()).unwrap_or_default();
    let errors = errors.lock().map(|e| e.clone()).unwrap_or_default();

    Ok(LoadReport {
        model: config.model,
        transport: client.transport().to_string(),
        concurrency,
        rows_per_request: config.rows_per_request.max(1),
        started_unix_ms,
        summary: Summary {
            completed,
            failed,
            wall_secs,
            throughput_rps: if wall_secs > 0.0 {
                completed as f64 / wall_secs
            } else {
                0.0
            },
            rows_per_sec: if wall_secs > 0.0 {
                total_rows as f64 / wall_secs
            } else {
                0.0
            },
            latency: Percentiles::of(latencies.clone()),
            errors,
        },
        timeseries: ticks.lock().map(|t| t.clone()).unwrap_or_default(),
        processes: 1,
        sockets: Some(sockets),
        samples_ms: config.record_samples.then_some(latencies),
    })
}

/// Sends the warmup requests, `concurrency` at a time.
///
/// Running them at the concurrency of the measured phase is what opens the
/// connections the run will use and loads the model before any of it is
/// timed. The first failure stops the warmup and is returned: a model that
/// cannot answer is not something to measure around.
async fn warm_up(
    client: &Arc<dyn PredictV2Client>,
    template: &InferRequest,
    requests: usize,
    concurrency: usize,
) -> Result<()> {
    let remaining = Arc::new(AtomicU64::new(requests as u64));
    let mut workers = Vec::with_capacity(concurrency.min(requests));

    for _ in 0..concurrency.min(requests) {
        let client = Arc::clone(client);
        let request = clone_request(template);
        let remaining = Arc::clone(&remaining);

        workers.push(tokio::spawn(async move {
            // Each worker claims a request before sending it, so the warmup
            // sends the count it was given however the work divides.
            while remaining
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |left| {
                    left.checked_sub(1)
                })
                .is_ok()
            {
                client.infer(clone_request(&request)).await?;
            }
            Ok::<(), InferError>(())
        }));
    }

    for worker in workers {
        match worker.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                return Err(InferError::InvalidRequest(format!(
                    "a warmup request failed, so there is nothing to measure: {e}"
                )))
            }
            Err(e) => return Err(InferError::Internal(format!("a warmup task failed: {e}"))),
        }
    }
    Ok(())
}

/// Clones a request. The template is reused rather than rebuilt per call so
/// that measurement covers the server, not tensor construction.
fn clone_request(template: &InferRequest) -> InferRequest {
    InferRequest {
        model: template.model.clone(),
        id: None,
        parameters: template.parameters.clone(),
        inputs: template.inputs.clone(),
        requested_outputs: template.requested_outputs.clone(),
    }
}

/// Records a tick per interval until the run stops.
async fn poll_metrics(
    endpoint: String,
    interval: Duration,
    started: Instant,
    counters: Arc<Counters>,
    ticks: Arc<Mutex<Vec<Tick>>>,
    model: String,
) {
    let mut ticker = tokio::time::interval(interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut previous: Option<(Duration, u64)> = None;

    loop {
        ticker.tick().await;
        let elapsed = started.elapsed();
        let completed = counters.completed.load(Ordering::Relaxed);

        // Throughput between ticks, which is what shows a ramp or a stall.
        let throughput_rps = match previous {
            Some((last_at, last_completed)) => {
                let gap = (elapsed - last_at).as_secs_f64();
                if gap > 0.0 {
                    (completed - last_completed) as f64 / gap
                } else {
                    0.0
                }
            }
            None => 0.0,
        };
        previous = Some((elapsed, completed));

        let mut tick = Tick {
            elapsed_ms: elapsed.as_millis() as u64,
            unix_ms: unix_ms(),
            completed,
            failed: counters.failed.load(Ordering::Relaxed),
            in_flight: counters.in_flight.load(Ordering::Relaxed),
            throughput_rps,
            ..Default::default()
        };

        // A metrics poll that fails must not stop the load, so the tick simply
        // carries no server-side readings.
        match scrape(&endpoint).await {
            Ok(scraped) => fill_server_readings(&mut tick, &scraped, &model),
            Err(e) => debug!(error = %e, "metrics poll failed"),
        }

        if let Ok(mut ticks) = ticks.lock() {
            ticks.push(tick);
        }
        if counters.should_stop() {
            return;
        }
    }
}

fn fill_server_readings(tick: &mut Tick, scraped: &Scrape, model: &str) {
    let for_model = |name: &str| scraped.value_where(name, &[("model", model)]);

    tick.cpu_utilization = scraped.value("gm_cpu_utilization");
    tick.memory_bytes = scraped.value("gm_memory_usage_bytes");
    tick.memory_utilization = scraped.value("gm_memory_utilization");
    tick.server_inference_count = for_model("gm_inference_count");
    tick.server_exec_count = for_model("gm_inference_exec_count");
    tick.server_pending = for_model("gm_inference_pending_request_count");
    tick.server_queue_duration_us = for_model("gm_inference_queue_duration_us");
    tick.server_compute_duration_us = for_model("gm_inference_compute_infer_duration_us");
    tick.mean_batch_size = match (tick.server_inference_count, tick.server_exec_count) {
        (Some(inferences), Some(executions)) if executions > 0.0 => Some(inferences / executions),
        _ => None,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    use galemind_core::{DataType, InferResponse, ModelMetadata, ServerMetadata, TensorMetadata};

    /// A client that answers instantly and counts what it was asked.
    struct CountingClient {
        inferences: AtomicU64,
    }

    impl CountingClient {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                inferences: AtomicU64::new(0),
            })
        }

        fn count(&self) -> u64 {
            self.inferences.load(Ordering::Relaxed)
        }
    }

    #[async_trait::async_trait]
    impl PredictV2Client for CountingClient {
        async fn server_live(&self) -> Result<bool> {
            Ok(true)
        }

        async fn server_ready(&self) -> Result<bool> {
            Ok(true)
        }

        async fn model_ready(&self, _model: &ModelRef) -> Result<bool> {
            Ok(true)
        }

        async fn server_metadata(&self) -> Result<ServerMetadata> {
            Ok(ServerMetadata {
                name: "test".to_string(),
                version: "0".to_string(),
                extensions: Vec::new(),
            })
        }

        async fn model_metadata(&self, model: &ModelRef) -> Result<ModelMetadata> {
            Ok(ModelMetadata {
                name: model.name.clone(),
                versions: vec!["1".to_string()],
                platform: "onnx_onnxv1".to_string(),
                inputs: vec![TensorMetadata {
                    name: "input".to_string(),
                    datatype: DataType::Fp32,
                    shape: vec![-1, 3],
                }],
                outputs: vec![TensorMetadata {
                    name: "output".to_string(),
                    datatype: DataType::Fp32,
                    shape: vec![-1, 1],
                }],
                max_batch_size: Some(32),
            })
        }

        async fn infer(&self, request: InferRequest) -> Result<InferResponse> {
            self.inferences.fetch_add(1, Ordering::Relaxed);
            Ok(InferResponse {
                model_name: request.model.name,
                model_version: "1".to_string(),
                id: None,
                parameters: Parameters::new(),
                outputs: Vec::new(),
            })
        }

        fn transport(&self) -> &'static str {
            "test"
        }
    }

    /// A bounded run must send the count it was given, not the count rounded
    /// up to a multiple of the concurrency.
    #[tokio::test]
    async fn a_bounded_run_sends_exactly_the_requests_it_was_asked_for() {
        // 100 requests across 64 workers is the awkward case: dividing the
        // work per worker would round up to 128.
        let client = CountingClient::new();
        let report = run_load(
            Arc::clone(&client) as Arc<dyn PredictV2Client>,
            LoadConfig {
                concurrency: 64,
                total_requests: Some(100),
                warmup_requests: 0,
                metrics_interval: None,
                metrics_endpoint: None,
                ..Default::default()
            },
        )
        .await
        .expect("the run completes");

        assert_eq!(client.count(), 100, "sent exactly the requests asked for");
        assert_eq!(report.summary.completed, 100);
        assert_eq!(report.summary.failed, 0);
        assert_eq!(report.summary.latency.min_ms > 0.0, true);
    }

    /// Warmup requests are sent, but must not land in the measurement.
    #[tokio::test]
    async fn warmup_requests_are_sent_but_not_measured() {
        let client = CountingClient::new();
        let report = run_load(
            Arc::clone(&client) as Arc<dyn PredictV2Client>,
            LoadConfig {
                concurrency: 4,
                total_requests: Some(20),
                warmup_requests: 5,
                metrics_interval: None,
                metrics_endpoint: None,
                ..Default::default()
            },
        )
        .await
        .expect("the run completes");

        assert_eq!(client.count(), 25, "20 measured and 5 warmup");
        assert_eq!(report.summary.completed, 20);
    }

    /// A warmup shorter than the concurrency would leave workers connecting
    /// while they were being measured, so it is raised to cover them.
    #[tokio::test]
    async fn the_warmup_covers_every_worker_the_run_will_use() {
        let client = CountingClient::new();
        let report = run_load(
            Arc::clone(&client) as Arc<dyn PredictV2Client>,
            LoadConfig {
                concurrency: 32,
                total_requests: Some(10),
                // Fewer than the concurrency: every worker still warms up.
                warmup_requests: 4,
                metrics_interval: None,
                metrics_endpoint: None,
                ..Default::default()
            },
        )
        .await
        .expect("the run completes");

        assert_eq!(client.count(), 42, "10 measured and 32 warmup");
        assert_eq!(report.summary.completed, 10);
    }

    /// Asking for no warmup must mean none, not one per worker.
    #[tokio::test]
    async fn a_warmup_can_still_be_turned_off_entirely() {
        let client = CountingClient::new();
        run_load(
            Arc::clone(&client) as Arc<dyn PredictV2Client>,
            LoadConfig {
                concurrency: 8,
                total_requests: Some(5),
                warmup_requests: 0,
                metrics_interval: None,
                metrics_endpoint: None,
                ..Default::default()
            },
        )
        .await
        .expect("the run completes");

        assert_eq!(client.count(), 5, "nothing but the measured requests");
    }

    /// A run bounded by neither a count nor a clock would never end, so it is
    /// refused before anything is sent.
    #[tokio::test]
    async fn an_unbounded_run_is_refused() {
        let client = CountingClient::new();
        let error = run_load(
            client.clone() as Arc<dyn PredictV2Client>,
            LoadConfig {
                total_requests: None,
                duration: None,
                ..Default::default()
            },
        )
        .await
        .expect_err("an unbounded run is refused");

        assert!(matches!(error, InferError::InvalidRequest(_)));
        assert_eq!(client.count(), 0, "nothing was sent");
    }

    #[test]
    fn percentiles_report_latencies_that_really_happened() {
        let latencies: Vec<f64> = (1..=100).map(|v| v as f64).collect();
        let p = Percentiles::of(latencies);

        assert_eq!(p.min_ms, 1.0);
        assert_eq!(p.p50_ms, 50.0);
        assert_eq!(p.p55_ms, 55.0);
        assert_eq!(p.p90_ms, 90.0);
        assert_eq!(p.p95_ms, 95.0);
        assert_eq!(p.p99_ms, 99.0);
        assert_eq!(p.max_ms, 100.0);
        assert_eq!(p.mean_ms, 50.5);
    }

    #[test]
    fn percentile_ranks_survive_binary_floating_point() {
        // Every tenth of a percent, over a hundred samples, must land on the
        // sample whose rank it names.
        let latencies: Vec<f64> = (1..=100).map(|v| v as f64).collect();
        for rank in 1..=100u32 {
            let quantile = rank as f64 / 100.0;
            let p = Percentiles::of(latencies.clone());
            let expected = rank as f64;
            let got = match rank {
                50 => p.p50_ms,
                55 => p.p55_ms,
                90 => p.p90_ms,
                95 => p.p95_ms,
                99 => p.p99_ms,
                100 => p.max_ms,
                _ => continue,
            };
            assert_eq!(got, expected, "quantile {quantile}");
        }
    }

    #[test]
    fn percentiles_are_order_independent() {
        let ascending = Percentiles::of(vec![1.0, 2.0, 3.0, 4.0]);
        let shuffled = Percentiles::of(vec![3.0, 1.0, 4.0, 2.0]);
        assert_eq!(ascending, shuffled);
    }

    #[test]
    fn a_single_sample_is_every_percentile() {
        let p = Percentiles::of(vec![7.5]);
        assert_eq!(p.p50_ms, 7.5);
        assert_eq!(p.p99_ms, 7.5);
        assert_eq!(p.max_ms, 7.5);
    }

    #[test]
    fn no_samples_gives_zeros_rather_than_a_panic() {
        assert_eq!(Percentiles::of(Vec::new()), Percentiles::default());
    }

    #[test]
    fn failures_are_grouped_by_kind() {
        assert_eq!(error_label(&InferError::Overloaded), "overloaded");
        assert_eq!(
            error_label(&InferError::Timeout(Duration::from_secs(1))),
            "timeout"
        );
        assert_eq!(
            error_label(&InferError::remote_client_error("x")),
            "remote_client_error"
        );
    }

    #[test]
    fn server_readings_are_taken_from_the_scrape_for_the_model_under_test() {
        let scraped = Scrape::parse(
            "gm_cpu_utilization 0.5\n\
             gm_memory_usage_bytes 1024\n\
             gm_inference_count{model=\"energy\",version=\"1\"} 40\n\
             gm_inference_exec_count{model=\"energy\",version=\"1\"} 10\n\
             gm_inference_count{model=\"other\",version=\"1\"} 999\n",
        );
        let mut tick = Tick::default();
        fill_server_readings(&mut tick, &scraped, "energy");

        assert_eq!(tick.cpu_utilization, Some(0.5));
        assert_eq!(tick.memory_bytes, Some(1024.0));
        assert_eq!(tick.server_inference_count, Some(40.0));
        assert_eq!(tick.mean_batch_size, Some(4.0), "40 rows over 10 batches");
    }

    #[test]
    fn a_scrape_without_batch_counts_reports_no_batch_size() {
        let mut tick = Tick::default();
        fill_server_readings(
            &mut tick,
            &Scrape::parse("gm_cpu_utilization 0\n"),
            "energy",
        );
        assert_eq!(tick.mean_batch_size, None);
    }

    #[test]
    fn a_report_serialises_to_json_a_timeseries_store_can_read() {
        let report = LoadReport {
            model: "energy".to_string(),
            transport: "http".to_string(),
            concurrency: 4,
            rows_per_request: 1,
            started_unix_ms: 1_700_000_000_000,
            summary: Summary {
                completed: 10,
                latency: Percentiles::of(vec![1.0, 2.0]),
                ..Default::default()
            },
            timeseries: vec![Tick {
                elapsed_ms: 500,
                completed: 5,
                cpu_utilization: Some(0.3),
                ..Default::default()
            }],
            processes: 1,
            sockets: Some(4),
            samples_ms: None,
        };

        let json = report.to_json();
        let parsed: LoadReport = serde_json::from_str(&json).expect("round trips");
        assert_eq!(parsed.summary.completed, 10);
        assert_eq!(parsed.timeseries[0].cpu_utilization, Some(0.3));
        assert_eq!(parsed.summary.latency.p50_ms, 1.0);
        assert_eq!(parsed.processes, 1);
    }

    /// A report from before `processes` existed still reads, as the oldest
    /// report in someone's timeseries store will be.
    #[test]
    fn a_report_without_the_newer_fields_still_reads() {
        let json = r#"{
            "model": "energy", "transport": "http", "concurrency": 4,
            "rows_per_request": 1, "started_unix_ms": 1,
            "summary": {"completed": 1, "failed": 0, "wall_secs": 1.0,
                        "throughput_rps": 1.0, "rows_per_sec": 1.0,
                        "latency": {"min_ms": 1.0, "p50_ms": 1.0, "p55_ms": 1.0,
                                    "p90_ms": 1.0, "p95_ms": 1.0, "p99_ms": 1.0,
                                    "max_ms": 1.0, "mean_ms": 1.0},
                        "errors": {}},
            "timeseries": []
        }"#;
        let parsed: LoadReport = serde_json::from_str(json).expect("still reads");
        assert_eq!(parsed.processes, 1);
        assert_eq!(parsed.sockets, None);
    }

    fn report_of(concurrency: usize, completed: u64, wall: f64, samples: &[f64]) -> LoadReport {
        LoadReport {
            model: "energy".to_string(),
            transport: "http".to_string(),
            concurrency,
            rows_per_request: 2,
            started_unix_ms: 1_700_000_000_000,
            summary: Summary {
                completed,
                failed: 1,
                wall_secs: wall,
                throughput_rps: completed as f64 / wall,
                rows_per_sec: (completed * 2) as f64 / wall,
                latency: Percentiles::of(samples.to_vec()),
                errors: BTreeMap::from([("overloaded".to_string(), 1)]),
            },
            timeseries: Vec::new(),
            processes: 1,
            sockets: Some(concurrency),
            samples_ms: Some(samples.to_vec()),
        }
    }

    /// What `--processes 10 -c 1000` has to add up to: ten thousand sockets
    /// and one set of numbers.
    #[test]
    fn the_reports_of_several_processes_add_up_to_one_run() {
        let reports = vec![
            report_of(1_000, 50_000, 10.0, &[1.0, 2.0, 3.0, 4.0]),
            report_of(1_000, 40_000, 8.0, &[5.0, 6.0, 7.0, 8.0]),
        ];
        let merged = LoadReport::merge(reports).expect("a merged report");

        assert_eq!(merged.processes, 2);
        // The sockets and the requests in flight are what the processes were
        // added together for.
        assert_eq!(merged.concurrency, 2_000);
        assert_eq!(merged.sockets, Some(2_000));
        assert_eq!(merged.summary.completed, 90_000);
        assert_eq!(merged.summary.failed, 2);
        assert_eq!(merged.summary.errors["overloaded"], 2);

        // Rates are against the longest wall time, because the processes
        // overlapped: 90 000 requests in the 10 seconds the slowest took.
        assert_eq!(merged.summary.wall_secs, 10.0);
        assert_eq!(merged.summary.throughput_rps, 9_000.0);
        // 100 000 + 80 000 rows over the same 10 seconds.
        assert_eq!(merged.summary.rows_per_sec, 18_000.0);

        // And the percentiles come from the pooled samples, not from an
        // average of percentiles: eight samples, 1..8.
        assert_eq!(merged.summary.latency.min_ms, 1.0);
        assert_eq!(merged.summary.latency.max_ms, 8.0);
        assert_eq!(merged.summary.latency.p50_ms, 4.0);
        assert_eq!(merged.summary.latency.mean_ms, 4.5);
    }

    #[test]
    fn merging_nothing_is_nothing_rather_than_a_report_of_zeroes() {
        assert!(LoadReport::merge(Vec::new()).is_none());
    }
}
