//! The per-model batching dispatcher.
//!
//! One dispatcher owns one model version. Requests arrive on a bounded queue;
//! the dispatcher collects them until the batch policy's size or window is
//! reached, runs them as a single call, and returns each caller its own rows.
//!
//! The queue is bounded on purpose. The structure this replaces was a ring
//! buffer that overwrote the oldest pending request once full, which silently
//! dropped work that a client was still waiting for. A full queue here is
//! reported as [`InferError::Overloaded`] so the caller learns the server is
//! saturated and can retry or shed load.

use crate::merge::{merge_requests, split_response};
use crate::policy::{BatchPolicy, TrafficMeter};
use galemind_core::{
    Backend, Clock, InferError, InferRequest, InferResponse, ModelCache, ModelKey,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tracing::{debug, trace, warn};

type Result<T> = std::result::Result<T, InferError>;

/// Queue and deadline settings for one dispatcher.
#[derive(Debug, Clone, Copy)]
pub struct DispatcherConfig {
    /// Requests that may wait at once. Beyond this the server reports itself
    /// overloaded rather than queueing without bound.
    pub queue_capacity: usize,
    /// A request waiting longer than this is failed instead of being run, so a
    /// backlog does not turn into work nobody is waiting for any more.
    pub queue_timeout: Duration,
    /// How many recent batches the traffic averages cover.
    pub sample_window: usize,
}

impl Default for DispatcherConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 1_024,
            queue_timeout: Duration::from_secs(10),
            sample_window: 64,
        }
    }
}

/// Counters describing one model's batching behaviour.
#[derive(Debug, Default)]
pub struct DispatcherMetrics {
    requests: AtomicU64,
    batches: AtomicU64,
    merged_batches: AtomicU64,
    rows: AtomicU64,
    rejected: AtomicU64,
    timed_out: AtomicU64,
}

/// A point-in-time read of [`DispatcherMetrics`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DispatcherSnapshot {
    pub requests: u64,
    pub batches: u64,
    /// Batches that actually combined more than one request.
    pub merged_batches: u64,
    pub rows: u64,
    /// Requests refused because the queue was full.
    pub rejected: u64,
    /// Requests failed because they waited past the queue timeout.
    pub timed_out: u64,
}

impl DispatcherMetrics {
    fn snapshot(&self) -> DispatcherSnapshot {
        DispatcherSnapshot {
            requests: self.requests.load(Ordering::Relaxed),
            batches: self.batches.load(Ordering::Relaxed),
            merged_batches: self.merged_batches.load(Ordering::Relaxed),
            rows: self.rows.load(Ordering::Relaxed),
            rejected: self.rejected.load(Ordering::Relaxed),
            timed_out: self.timed_out.load(Ordering::Relaxed),
        }
    }
}

struct Job {
    request: InferRequest,
    enqueued_millis: u64,
    reply: oneshot::Sender<Result<InferResponse>>,
}

/// Accepts requests for one model and runs them in batches.
pub struct Dispatcher {
    key: ModelKey,
    sender: mpsc::Sender<Job>,
    metrics: Arc<DispatcherMetrics>,
    worker: JoinHandle<()>,
}

impl Dispatcher {
    /// Starts the dispatcher task for `key`.
    ///
    /// The backend is taken from the cache for each batch rather than held, so
    /// an idle model can still be unloaded and a replaced one is picked up.
    pub fn spawn(
        key: ModelKey,
        cache: Arc<dyn ModelCache>,
        policy: Arc<dyn BatchPolicy>,
        clock: Arc<dyn Clock>,
        config: DispatcherConfig,
    ) -> Self {
        let (sender, receiver) = mpsc::channel(config.queue_capacity.max(1));
        let metrics = Arc::new(DispatcherMetrics::default());

        let worker = tokio::spawn(run(
            key.clone(),
            receiver,
            cache,
            policy,
            clock,
            config,
            Arc::clone(&metrics),
        ));

        Self {
            key,
            sender,
            metrics,
            worker,
        }
    }

    pub fn model(&self) -> &ModelKey {
        &self.key
    }

    pub fn metrics(&self) -> DispatcherSnapshot {
        self.metrics.snapshot()
    }

    /// Submits a request and waits for its own rows of the batch result.
    pub async fn submit(&self, request: InferRequest, now_millis: u64) -> Result<InferResponse> {
        let (reply, response) = oneshot::channel();
        let job = Job {
            request,
            enqueued_millis: now_millis,
            reply,
        };

        self.metrics.requests.fetch_add(1, Ordering::Relaxed);
        self.sender.try_send(job).map_err(|e| match e {
            mpsc::error::TrySendError::Full(_) => {
                self.metrics.rejected.fetch_add(1, Ordering::Relaxed);
                debug!(model = %self.key, "queue full, refusing request");
                InferError::Overloaded
            }
            mpsc::error::TrySendError::Closed(_) => {
                InferError::Internal(format!("dispatcher for {} has stopped", self.key))
            }
        })?;

        response.await.map_err(|_| {
            InferError::Internal(format!("dispatcher for {} dropped a request", self.key))
        })?
    }

    /// Stops accepting requests and waits for the queue to drain.
    pub async fn shutdown(self) {
        drop(self.sender);
        let _ = self.worker.await;
    }
}

async fn run(
    key: ModelKey,
    mut receiver: mpsc::Receiver<Job>,
    cache: Arc<dyn ModelCache>,
    policy: Arc<dyn BatchPolicy>,
    clock: Arc<dyn Clock>,
    config: DispatcherConfig,
    metrics: Arc<DispatcherMetrics>,
) {
    let mut meter = TrafficMeter::new(config.sample_window);

    while let Some(first) = receiver.recv().await {
        meter.record_arrival(clock.now_millis());
        let mut batch = vec![first];

        // The backend is needed before the batch is sized: its metadata says
        // how large a batch the model permits and whether it can batch at all.
        let backend = match cache.get(&key).await {
            Ok(backend) => backend,
            Err(e) => {
                reply_all(batch, &e);
                continue;
            }
        };
        let model_max = backend.metadata().max_batch_size.unwrap_or(1).max(1);

        let stats = meter.stats(receiver.len(), model_max);
        let max_batch = policy.max_batch(&stats).max(1);
        let window = policy.window(&stats);
        trace!(model = %key, ?window, max_batch, queue = receiver.len(), "collecting batch");

        collect(
            &mut receiver,
            &mut batch,
            max_batch,
            window,
            &mut meter,
            &clock,
        )
        .await;

        let kept = drop_expired(batch, &clock, config.queue_timeout, &metrics);
        if kept.is_empty() {
            continue;
        }

        let rows = execute(&key, kept, backend, &metrics).await;
        meter.record_batch(rows.0, rows.1);
    }

    debug!(model = %key, "dispatcher stopped");
}

/// Collects further requests until the batch is full or the window closes.
async fn collect(
    receiver: &mut mpsc::Receiver<Job>,
    batch: &mut Vec<Job>,
    max_batch: usize,
    window: Duration,
    meter: &mut TrafficMeter,
    clock: &Arc<dyn Clock>,
) {
    // Whatever is already queued costs nothing to take.
    while batch.len() < max_batch {
        match receiver.try_recv() {
            Ok(job) => {
                meter.record_arrival(clock.now_millis());
                batch.push(job);
            }
            Err(_) => break,
        }
    }

    if window.is_zero() || batch.len() >= max_batch {
        return;
    }

    let deadline = Instant::now() + window;
    while batch.len() < max_batch {
        match tokio::time::timeout_at(deadline.into(), receiver.recv()).await {
            Ok(Some(job)) => {
                meter.record_arrival(clock.now_millis());
                batch.push(job);
            }
            // The channel closed: no more requests will ever arrive.
            Ok(None) => break,
            // The window closed: run with what we have.
            Err(_) => break,
        }
    }
}

/// Fails requests that waited past the queue timeout.
fn drop_expired(
    batch: Vec<Job>,
    clock: &Arc<dyn Clock>,
    timeout: Duration,
    metrics: &DispatcherMetrics,
) -> Vec<Job> {
    if timeout.is_zero() {
        return batch;
    }
    let now = clock.now_millis();
    let limit = timeout.as_millis() as u64;

    batch
        .into_iter()
        .filter_map(|job| {
            if now.saturating_sub(job.enqueued_millis) > limit {
                metrics.timed_out.fetch_add(1, Ordering::Relaxed);
                let _ = job.reply.send(Err(InferError::Timeout(timeout)));
                None
            } else {
                Some(job)
            }
        })
        .collect()
}

/// Runs a batch and answers every request in it. Returns the rows run and how
/// long they took, for the traffic averages.
async fn execute(
    key: &ModelKey,
    batch: Vec<Job>,
    backend: Arc<dyn Backend>,
    metrics: &DispatcherMetrics,
) -> (usize, Duration) {
    metrics.batches.fetch_add(1, Ordering::Relaxed);
    let started = Instant::now();

    // A model without a dynamic batch dimension cannot have requests merged,
    // so they are run one after another instead.
    if batch.len() == 1 || !backend.metadata().is_batchable() {
        let mut rows = 0;
        for job in batch {
            let (request, reply) = (job.request, job.reply);
            rows += request.batch_rows().unwrap_or(1);
            let result = run_blocking(key, Arc::clone(&backend), request).await;
            let _ = reply.send(result);
        }
        metrics.rows.fetch_add(rows as u64, Ordering::Relaxed);
        return (rows, started.elapsed());
    }

    metrics.merged_batches.fetch_add(1, Ordering::Relaxed);
    let requests: Vec<_> = batch.iter().map(|job| job.request.clone()).collect();
    let (merged, rows_per_request) = match merge_requests(&requests) {
        Ok(merged) => merged,
        Err(e) => {
            warn!(model = %key, error = %e, "falling back to one-by-one execution");
            reply_all(batch, &e);
            return (0, started.elapsed());
        }
    };
    let rows: usize = rows_per_request.iter().sum();

    let result = run_blocking(key, Arc::clone(&backend), merged).await;
    let elapsed = started.elapsed();
    metrics.rows.fetch_add(rows as u64, Ordering::Relaxed);

    match result {
        Err(e) => reply_all(batch, &e),
        Ok(response) => match split_response(&response, &rows_per_request) {
            Err(e) => reply_all(batch, &e),
            Ok(parts) => {
                debug!(model = %key, requests = batch.len(), rows, ?elapsed, "ran batch");
                for (job, mut part) in batch.into_iter().zip(parts) {
                    part.id = job.request.id.clone();
                    let _ = job
                        .reply
                        .send(Ok(part.select_outputs(&job.request.requested_outputs)));
                }
            }
        },
    }

    (rows, elapsed)
}

/// Runs one call on a blocking thread, since inference is CPU bound.
async fn run_blocking(
    key: &ModelKey,
    backend: Arc<dyn Backend>,
    request: InferRequest,
) -> Result<InferResponse> {
    let requested = request.requested_outputs.clone();
    tokio::task::spawn_blocking(move || backend.infer(&request))
        .await
        .map_err(|e| InferError::Internal(format!("inference task for {key} failed: {e}")))?
        .map(|response| response.select_outputs(&requested))
}

fn reply_all(batch: Vec<Job>, error: &InferError) {
    for job in batch {
        let _ = job.reply.send(Err(error.clone()));
    }
}
