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
    fn stats(&self) -> CacheStats {
        CacheStats::default()
    }
}

struct Harness {
    dispatcher: Dispatcher,
    calls: Arc<AtomicUsize>,
    largest_call_rows: Arc<AtomicUsize>,
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
    let backend = Arc::new(DoublingBackend {
        metadata: DoublingBackend::metadata(batchable),
        calls: Arc::clone(&calls),
        largest_call_rows: Arc::clone(&largest_call_rows),
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
        clock,
    }
}

/// A request whose single row is `[seed, seed + 1]`.
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
