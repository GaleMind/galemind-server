//! Behaviour of the per-model batching dispatcher.

use async_trait::async_trait;
use galemind_batching::{Dispatcher, DispatcherConfig, FixedBatchPolicy};
use galemind_core::{
    Backend, CacheStats, Clock, DataType, InferError, InferRequest, InferResponse, ModelCache,
    ModelKey, ModelMetadata, ModelRef, Parameters, RequestedOutput, SystemClock, Tensor,
    TensorData, TensorMetadata, TestClock, PLATFORM_ONNX,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// A backend that doubles every feature, so each caller's rows are identifiable
/// in the result and mis-routed rows are impossible to miss.
struct DoublingBackend {
    metadata: ModelMetadata,
    calls: Arc<AtomicUsize>,
    largest_call_rows: Arc<AtomicUsize>,
    /// Executions running right now, and the most there have ever been.
    ///
    /// The second is what says whether the dispatcher overlaps batches at
    /// all: a counter of calls cannot tell concurrency from a queue.
    in_flight: Arc<AtomicUsize>,
    most_in_flight: Arc<AtomicUsize>,
    delay: Duration,
    fail: bool,
}

impl DoublingBackend {
    fn metadata(batchable: bool) -> ModelMetadata {
        let batch_dim = if batchable { -1 } else { 1 };
        ModelMetadata {
            name: "doubler".to_string(),
            versions: vec!["1".to_string()],
            platform: PLATFORM_ONNX.to_string(),
            inputs: vec![TensorMetadata {
                name: "features".to_string(),
                datatype: DataType::Fp32,
                shape: vec![batch_dim, 2],
            }],
            outputs: vec![
                TensorMetadata {
                    name: "doubled".to_string(),
                    datatype: DataType::Fp32,
                    shape: vec![batch_dim, 2],
                },
                TensorMetadata {
                    name: "rows".to_string(),
                    datatype: DataType::Int64,
                    shape: vec![batch_dim],
                },
            ],
            max_batch_size: Some(if batchable { 32 } else { 1 }),
        }
    }
}

impl Backend for DoublingBackend {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn infer(&self, request: &InferRequest) -> Result<InferResponse, InferError> {
        let running = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.most_in_flight.fetch_max(running, Ordering::SeqCst);
        // Decremented however this returns, including on the failure paths
        // below, so the high-water mark means what it says.
        let _leave = LeaveOnDrop(Arc::clone(&self.in_flight));

        if !self.delay.is_zero() {
            std::thread::sleep(self.delay);
        }
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.fail {
            return Err(InferError::Backend("the backend is broken".to_string()));
        }

        let input = &request.inputs[0];
        let rows = input.rows().unwrap();
        self.largest_call_rows.fetch_max(rows, Ordering::SeqCst);

        let TensorData::Fp32(values) = &input.data else {
            panic!("expected FP32 input");
        };
        Ok(InferResponse {
            model_name: "doubler".to_string(),
            model_version: "1".to_string(),
            id: request.id.clone(),
            parameters: Parameters::new(),
            outputs: vec![
                Tensor::new(
                    "doubled",
                    input.shape.clone(),
                    TensorData::Fp32(values.iter().map(|v| v * 2.0).collect()),
                ),
                Tensor::new(
                    "rows",
                    vec![rows as i64],
                    TensorData::Int64(vec![rows as i64; rows]),
                ),
            ],
        })
    }

    fn size_bytes(&self) -> usize {
        1
    }
}

/// Decrements the in-flight count when an execution leaves `infer`.
struct LeaveOnDrop(Arc<AtomicUsize>);

impl Drop for LeaveOnDrop {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// A cache that always hands out the same backend.
struct StaticCache {
    backend: Arc<dyn Backend>,
    failure: Option<InferError>,
}

#[async_trait]
impl ModelCache for StaticCache {
    async fn get(&self, _key: &ModelKey) -> Result<Arc<dyn Backend>, InferError> {
        match &self.failure {
            Some(e) => Err(e.clone()),
            None => Ok(Arc::clone(&self.backend)),
        }
    }
    async fn invalidate(&self, _key: &ModelKey) {}
    async fn sweep(&self) -> usize {
        0
    }
    fn loaded_keys(&self) -> Vec<ModelKey> {
        // This cache holds no keys of its own: it answers for any model.
        Vec::new()
    }
    fn stats(&self) -> CacheStats {
        CacheStats::default()
    }
}

struct Harness {
    dispatcher: Dispatcher,
    calls: Arc<AtomicUsize>,
    largest_call_rows: Arc<AtomicUsize>,
    most_in_flight: Arc<AtomicUsize>,
    clock: Arc<TestClock>,
}

fn harness(
    batchable: bool,
    window: Duration,
    max_batch: usize,
    config: DispatcherConfig,
    delay: Duration,
    fail: bool,
) -> Harness {
    let calls = Arc::new(AtomicUsize::new(0));
    let largest_call_rows = Arc::new(AtomicUsize::new(0));
    let most_in_flight = Arc::new(AtomicUsize::new(0));
    let backend = Arc::new(DoublingBackend {
        metadata: DoublingBackend::metadata(batchable),
        calls: Arc::clone(&calls),
        largest_call_rows: Arc::clone(&largest_call_rows),
        in_flight: Arc::new(AtomicUsize::new(0)),
        most_in_flight: Arc::clone(&most_in_flight),
        delay,
        fail,
    });
    let cache = Arc::new(StaticCache {
        backend: backend as Arc<dyn Backend>,
        failure: None,
    });
    let clock = Arc::new(TestClock::new(0));

    Harness {
        dispatcher: Dispatcher::spawn(
            ModelKey::new("doubler", "1"),
            cache,
            Arc::new(FixedBatchPolicy { window, max_batch }),
            Arc::clone(&clock) as Arc<dyn Clock>,
            config,
        ),
        calls,
        largest_call_rows,
        most_in_flight,
        clock,
    }
}

/// A request of `rows` rows, the `r`th being `[seed + r, seed + r + 0.5]`.
fn request(seed: f32, rows: usize) -> InferRequest {
    let values: Vec<f32> = (0..rows)
        .flat_map(|r| [seed + r as f32, seed + r as f32 + 0.5])
        .collect();
    InferRequest {
        model: ModelRef::latest("doubler"),
        id: Some(format!("req-{seed}")),
        parameters: Parameters::new(),
        inputs: vec![Tensor::new(
            "features",
            vec![rows as i64, 2],
            TensorData::Fp32(values),
        )],
        requested_outputs: Vec::new(),
    }
}

fn doubled(response: &InferResponse) -> Vec<f32> {
    let output = response
        .outputs
        .iter()
        .find(|o| o.name == "doubled")
        .expect("doubled output");
    let TensorData::Fp32(values) = &output.data else {
        panic!("expected FP32");
    };
    values.clone()
}

#[tokio::test]
async fn a_lone_request_is_answered_with_its_own_rows() {
    let h = harness(
        true,
        Duration::ZERO,
        8,
        DispatcherConfig::default(),
        Duration::ZERO,
        false,
    );

    let response = h.dispatcher.submit(request(1.0, 2), 0).await.unwrap();

    assert_eq!(response.id.as_deref(), Some("req-1"));
    assert_eq!(response.model_name, "doubler");
    assert_eq!(doubled(&response), vec![2.0, 3.0, 4.0, 5.0]);
    assert_eq!(h.calls.load(Ordering::SeqCst), 1);

    let metrics = h.dispatcher.metrics();
    assert_eq!(metrics.requests, 1);
    assert_eq!(metrics.batches, 1);
    assert_eq!(metrics.merged_batches, 0, "one request is not a merge");
    assert_eq!(metrics.rows, 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_requests_are_merged_and_each_gets_back_only_its_own_rows() {
    let h = Arc::new(harness(
        true,
        Duration::from_millis(40),
        32,
        DispatcherConfig::default(),
        Duration::ZERO,
        false,
    ));

    let submissions: Vec<_> = (0..24)
        .map(|i| {
            let h = Arc::clone(&h);
            tokio::spawn(async move {
                let seed = (i * 10) as f32;
                let response = h.dispatcher.submit(request(seed, 1), 0).await.unwrap();
                (seed, response)
            })
        })
        .collect();

    for submission in submissions {
        let (seed, response) = submission.await.unwrap();
        assert_eq!(response.id.as_deref(), Some(format!("req-{seed}").as_str()));
        assert_eq!(
            doubled(&response),
            vec![seed * 2.0, (seed + 0.5) * 2.0],
            "request {seed} received another request's rows"
        );
    }

    let metrics = h.dispatcher.metrics();
    assert_eq!(metrics.requests, 24);
    assert_eq!(metrics.rows, 24);
    assert!(
        metrics.batches < 24,
        "requests should have shared batches, got {metrics:?}"
    );
    assert!(metrics.merged_batches >= 1, "{metrics:?}");
    assert!(
        h.largest_call_rows.load(Ordering::SeqCst) > 1,
        "at least one call should have carried several rows"
    );
}

#[tokio::test]
async fn a_model_without_a_dynamic_batch_dimension_is_never_merged() {
    let h = harness(
        false,
        Duration::from_millis(20),
        8,
        DispatcherConfig::default(),
        Duration::ZERO,
        false,
    );

    for seed in 0..4 {
        let response = h
            .dispatcher
            .submit(request(seed as f32, 1), 0)
            .await
            .unwrap();
        assert_eq!(doubled(&response).len(), 2);
    }

    assert_eq!(h.calls.load(Ordering::SeqCst), 4);
    assert_eq!(h.dispatcher.metrics().merged_batches, 0);
    assert_eq!(h.largest_call_rows.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn only_the_requested_outputs_come_back() {
    let h = harness(
        true,
        Duration::ZERO,
        8,
        DispatcherConfig::default(),
        Duration::ZERO,
        false,
    );
    let mut request = request(3.0, 1);
    request.requested_outputs = vec![RequestedOutput {
        name: "rows".to_string(),
        parameters: Parameters::new(),
    }];

    let response = h.dispatcher.submit(request, 0).await.unwrap();
    assert_eq!(response.outputs.len(), 1);
    assert_eq!(response.outputs[0].name, "rows");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_full_queue_reports_the_server_as_overloaded_rather_than_dropping_work() {
    let h = Arc::new(harness(
        true,
        Duration::ZERO,
        1,
        DispatcherConfig {
            queue_capacity: 1,
            ..DispatcherConfig::default()
        },
        Duration::from_millis(60),
        false,
    ));

    let submissions: Vec<_> = (0..32)
        .map(|i| {
            let h = Arc::clone(&h);
            tokio::spawn(async move { h.dispatcher.submit(request(i as f32, 1), 0).await })
        })
        .collect();

    let mut overloaded = 0;
    let mut served = 0;
    for submission in submissions {
        match submission.await.unwrap() {
            Ok(_) => served += 1,
            Err(InferError::Overloaded) => overloaded += 1,
            Err(e) => panic!("unexpected failure: {e}"),
        }
    }

    assert!(overloaded > 0, "a saturated queue must refuse work");
    assert!(served > 0, "the queue must still serve what it accepted");
    assert_eq!(served + overloaded, 32);
    assert_eq!(h.dispatcher.metrics().rejected, overloaded);
}

#[tokio::test]
async fn a_request_that_waited_past_the_queue_timeout_is_failed_not_run() {
    let h = harness(
        true,
        Duration::ZERO,
        8,
        DispatcherConfig {
            queue_timeout: Duration::from_secs(5),
            ..DispatcherConfig::default()
        },
        Duration::ZERO,
        false,
    );

    // The request is stamped as having arrived long before the clock now reads.
    h.clock.advance_secs(30);
    let result = h.dispatcher.submit(request(1.0, 1), 0).await;

    assert!(matches!(result, Err(InferError::Timeout(_))), "{result:?}");
    assert_eq!(h.calls.load(Ordering::SeqCst), 0, "no work should have run");
    assert_eq!(h.dispatcher.metrics().timed_out, 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_failing_backend_reports_the_failure_to_every_request_in_the_batch() {
    let h = Arc::new(harness(
        true,
        Duration::from_millis(30),
        32,
        DispatcherConfig::default(),
        Duration::ZERO,
        true,
    ));

    let submissions: Vec<_> = (0..8)
        .map(|i| {
            let h = Arc::clone(&h);
            tokio::spawn(async move { h.dispatcher.submit(request(i as f32, 1), 0).await })
        })
        .collect();

    for submission in submissions {
        let result = submission.await.unwrap();
        assert!(matches!(result, Err(InferError::Backend(_))), "{result:?}");
    }
}

#[tokio::test]
async fn a_model_that_cannot_be_loaded_fails_the_request() {
    let cache = Arc::new(StaticCache {
        backend: Arc::new(DoublingBackend {
            metadata: DoublingBackend::metadata(true),
            calls: Arc::new(AtomicUsize::new(0)),
            largest_call_rows: Arc::new(AtomicUsize::new(0)),
            in_flight: Arc::new(AtomicUsize::new(0)),
            most_in_flight: Arc::new(AtomicUsize::new(0)),
            delay: Duration::ZERO,
            fail: false,
        }),
        failure: Some(InferError::ModelNotFound("doubler".to_string())),
    });
    let dispatcher = Dispatcher::spawn(
        ModelKey::new("doubler", "1"),
        cache,
        Arc::new(FixedBatchPolicy {
            window: Duration::ZERO,
            max_batch: 4,
        }),
        Arc::new(SystemClock::new()),
        DispatcherConfig::default(),
    );

    let result = dispatcher.submit(request(1.0, 1), 0).await;
    assert!(
        matches!(result, Err(InferError::ModelNotFound(_))),
        "{result:?}"
    );
    dispatcher.shutdown().await;
}

#[tokio::test]
async fn requests_submitted_after_shutdown_are_refused() {
    let h = harness(
        true,
        Duration::ZERO,
        4,
        DispatcherConfig::default(),
        Duration::ZERO,
        false,
    );
    h.dispatcher.submit(request(1.0, 1), 0).await.unwrap();
    h.dispatcher.shutdown().await;
}

/// The dispatcher runs several batches at once, which is what lets one hot
/// model use more than one core.
///
/// Measured as overlap rather than as throughput: the backend reports the
/// most executions it ever had running at the same time, and a dispatcher
/// that awaited each batch before forming the next could never report more
/// than one however fast it went.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn batches_of_one_model_execute_concurrently() {
    // A batch of one request, so eight requests are eight batches rather
    // than one merged call -- it is batches that are being overlapped here.
    let harness = harness(
        true,
        Duration::ZERO,
        1,
        DispatcherConfig {
            max_concurrent_batches: 4,
            ..DispatcherConfig::default()
        },
        Duration::from_millis(60),
        false,
    );

    let harness = Arc::new(harness);
    let submissions: Vec<_> = (0..8)
        .map(|seed| {
            let harness = Arc::clone(&harness);
            tokio::spawn(async move { harness.dispatcher.submit(request(seed as f32, 1), 0).await })
        })
        .collect();
    for submission in submissions {
        submission
            .await
            .expect("the task")
            .expect("every request is answered");
    }

    let most = harness.most_in_flight.load(Ordering::SeqCst);
    assert!(
        most > 1,
        "the dispatcher ran one batch at a time: a model can never use more \
         than one core (most in flight: {most})"
    );
    assert!(
        most <= 4,
        "the per-model bound was exceeded: {most} batches at once"
    );
    assert_eq!(harness.calls.load(Ordering::SeqCst), 8);
}

/// And the bound is a bound: at one, the old serial behaviour is still
/// available for a deployment that wants the smallest possible working set.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn one_concurrent_batch_is_still_one_at_a_time() {
    let harness = harness(
        true,
        Duration::ZERO,
        1,
        DispatcherConfig {
            max_concurrent_batches: 1,
            ..DispatcherConfig::default()
        },
        Duration::from_millis(20),
        false,
    );

    let harness = Arc::new(harness);
    let submissions: Vec<_> = (0..6)
        .map(|seed| {
            let harness = Arc::clone(&harness);
            tokio::spawn(async move { harness.dispatcher.submit(request(seed as f32, 1), 0).await })
        })
        .collect();
    for submission in submissions {
        submission
            .await
            .expect("the task")
            .expect("every request is answered");
    }

    assert_eq!(
        harness.most_in_flight.load(Ordering::SeqCst),
        1,
        "a bound of one must serialise"
    );
    assert_eq!(harness.calls.load(Ordering::SeqCst), 6);
}

/// Concurrency must not cost anyone their own answer: with batches
/// overlapping and completing out of order, each caller still gets the rows
/// it sent and nobody else's.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_batches_still_answer_each_caller_with_its_own_rows() {
    let harness = harness(
        true,
        Duration::ZERO,
        1,
        DispatcherConfig {
            max_concurrent_batches: 4,
            ..DispatcherConfig::default()
        },
        // Uneven delays, so completions interleave rather than finishing in
        // the order they started.
        Duration::from_millis(5),
        false,
    );

    let harness = Arc::new(harness);
    let submissions: Vec<_> = (0..12)
        .map(|seed| {
            let harness = Arc::clone(&harness);
            tokio::spawn(async move {
                let response = harness
                    .dispatcher
                    .submit(request(seed as f32, 1), 0)
                    .await
                    .expect("the backend succeeded");
                (seed, response)
            })
        })
        .collect();
    for submission in submissions {
        let (seed, response) = submission.await.expect("the task");
        assert_eq!(
            doubled(&response),
            vec![seed as f32 * 2.0, seed as f32 * 2.0 + 1.0],
            "request {seed} was given another request's rows"
        );
    }
}
