//! The client-to-server flow, end to end, against a real server.
//!
//! These tests start the published container image and talk to it with the
//! shipped client, over both transports, for each kind of model in the
//! repository: a regression that returns one number per row, and a classifier
//! that returns a label and a row of probabilities. What is being tested is
//! the flow as a user meets it — image, server, protocol, client — so nothing
//! here reaches inside the server's own crates.
//!
//! Run with `cargo e2e`. Docker must be running, and the image must exist:
//!
//! ```text
//! docker compose build server
//! cargo e2e
//! ```

use galemind_client::{connect, scrape, PredictV2Client, Transport};
use galemind_core::{DataType, InferRequest, ModelRef, Parameters, Tensor, TensorData};
use std::time::Duration;
use testcontainers::{
    core::{IntoContainerPort, WaitFor},
    runners::AsyncRunner,
    ContainerAsync, GenericImage, ImageExt,
};
use tokio::sync::OnceCell;

/// The image the tests drive. `docker compose build server` produces it.
const IMAGE: &str = "galemind-server";
const TAG: &str = "0.2.0";

const REGRESSION: &str = "building_energy";
const CLASSIFIER: &str = "building_energy_class";

/// The eight features both models take: floor area, occupancy, outdoor
/// temperature, hour of day, HVAC setpoint, insulation rating, weekend flag,
/// solar irradiance.
const BIG_OFFICE: [f32; 8] = [1800.0, 30.0, -4.0, 18.0, 22.0, 1.0, 0.0, 40.0];
const SMALL_FLAT: [f32; 8] = [55.0, 2.0, 18.0, 14.0, 21.0, 5.0, 1.0, 600.0];

/// One server, shared by every test in the file. Starting a container per test
/// would multiply a slow step by the number of tests for no added coverage,
/// and the suite runs single-threaded (see the `e2e` alias) so the metrics
/// assertions are not disturbed by a concurrent test.
static SERVER: OnceCell<Server> = OnceCell::const_new();

struct Server {
    /// Held so the container outlives the tests; dropping it stops the server.
    _container: ContainerAsync<GenericImage>,
    rest: String,
    grpc: String,
}

async fn server() -> &'static Server {
    SERVER
        .get_or_init(|| async {
            let container = GenericImage::new(IMAGE, TAG)
                .with_exposed_port(8000.tcp())
                .with_exposed_port(8001.tcp())
                // The line the server logs once both listeners are up. The
                // readiness poll below is what the tests actually rely on;
                // this only avoids hammering a port that is not open yet.
                .with_wait_for(WaitFor::message_on_stdout("serving Prometheus metrics"))
                // The cache has to cover the repository for --preload to
                // leave everything loaded: the repository holds eleven
                // versions and the default capacity is eight, which would
                // evict the first three as the last three loaded.
                .with_cmd([
                    "serve",
                    "--preload",
                    "--cache-capacity=16",
                    "--metrics-interval-ms=200",
                ])
                .start()
                .await
                .expect(
                    "cannot start the server image; run `docker compose build server` first, \
                     and check that Docker is running",
                );

            let host = container.get_host().await.expect("container host");
            let rest = format!(
                "http://{host}:{}",
                container
                    .get_host_port_ipv4(8000.tcp())
                    .await
                    .expect("REST port")
            );
            let grpc = format!(
                "http://{host}:{}",
                container
                    .get_host_port_ipv4(8001.tcp())
                    .await
                    .expect("gRPC port")
            );

            let server = Server {
                _container: container,
                rest,
                grpc,
            };
            server.await_ready().await;
            server
        })
        .await
}

impl Server {
    /// Waits until the server reports every model ready. `--preload` loads the
    /// whole repository before reporting readiness, so this also means the
    /// models are in memory and no test pays for a first load.
    async fn await_ready(&self) {
        let client = connect(Transport::Http, &self.rest)
            .await
            .expect("an HTTP client");

        for _ in 0..100 {
            if matches!(client.server_ready().await, Ok(true)) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        panic!("the server did not become ready within 20 seconds");
    }

    async fn client(&self, transport: Transport) -> Box<dyn PredictV2Client> {
        let endpoint = match transport {
            // HTTP/2 is the REST port, framed differently.
            Transport::Http | Transport::Http2 => &self.rest,
            Transport::Grpc => &self.grpc,
        };
        connect(transport, endpoint)
            .await
            .expect("a client for the transport")
    }
}

/// Builds a request carrying `rows` of the eight features.
fn request(model: &str, rows: &[[f32; 8]]) -> InferRequest {
    let data: Vec<f32> = rows.iter().flatten().copied().collect();
    InferRequest {
        model: ModelRef::new(model.to_string(), None),
        id: None,
        parameters: Parameters::new(),
        inputs: vec![Tensor::new(
            "features",
            vec![rows.len() as i64, 8],
            TensorData::Fp32(data),
        )],
        requested_outputs: Vec::new(),
    }
}

/// Reads an FP32 tensor's values, failing the test if it is some other type.
fn fp32(tensor: &Tensor) -> Vec<f32> {
    match &tensor.data {
        TensorData::Fp32(values) => values.clone(),
        other => panic!("output '{}' is {}, not FP32", tensor.name, other.datatype()),
    }
}

/// Reads an INT64 tensor's values, failing the test if it is some other type.
fn int64(tensor: &Tensor) -> Vec<i64> {
    match &tensor.data {
        TensorData::Int64(values) => values.clone(),
        other => panic!(
            "output '{}' is {}, not INT64",
            tensor.name,
            other.datatype()
        ),
    }
}

/// Both transports must answer the protocol's own three questions the same
/// way, since a load balancer in front of either reads them.
#[tokio::test]
async fn the_server_reports_itself_live_ready_and_names_its_extensions() {
    let server = server().await;

    for transport in [Transport::Http, Transport::Grpc] {
        let client = server.client(transport).await;

        assert!(
            client.server_live().await.expect("live answers"),
            "{transport} reports the server live"
        );
        assert!(
            client.server_ready().await.expect("ready answers"),
            "{transport} reports the server ready"
        );

        let metadata = client.server_metadata().await.expect("server metadata");
        assert_eq!(metadata.name, "galemind", "{transport} names the server");
        assert!(
            !metadata.version.is_empty(),
            "{transport} reports a version"
        );
    }
}

/// The regression model: one input of eight features, one output of one number
/// per row, batchable.
#[tokio::test]
async fn the_regression_model_describes_itself_over_both_transports() {
    let server = server().await;

    for transport in [Transport::Http, Transport::Grpc] {
        let client = server.client(transport).await;
        let model = ModelRef::new(REGRESSION.to_string(), None);

        assert!(
            client.model_ready(&model).await.expect("ready answers"),
            "{transport}: {REGRESSION} is ready"
        );

        let metadata = client.model_metadata(&model).await.expect("model metadata");
        assert_eq!(metadata.name, REGRESSION);
        assert_eq!(metadata.platform, "onnx_onnxv1");
        assert_eq!(metadata.versions, vec!["1".to_string()]);
        assert_eq!(metadata.max_batch_size, Some(32));

        assert_eq!(metadata.inputs.len(), 1, "{transport}: one input");
        assert_eq!(metadata.inputs[0].name, "features");
        assert_eq!(metadata.inputs[0].datatype, DataType::Fp32);
        assert_eq!(
            metadata.inputs[0].shape,
            vec![-1, 8],
            "{transport}: a dynamic batch of eight features"
        );

        assert_eq!(metadata.outputs.len(), 1, "{transport}: one output");
        assert_eq!(metadata.outputs[0].name, "kwh");
        assert_eq!(metadata.outputs[0].datatype, DataType::Fp32);
        assert_eq!(metadata.outputs[0].shape, vec![-1, 1]);
    }
}

/// The classifier: the same input, but a label and a row of class
/// probabilities out, and the label is an integer rather than a float.
#[tokio::test]
async fn the_classifier_describes_itself_over_both_transports() {
    let server = server().await;

    for transport in [Transport::Http, Transport::Grpc] {
        let client = server.client(transport).await;
        let model = ModelRef::new(CLASSIFIER.to_string(), None);

        let metadata = client.model_metadata(&model).await.expect("model metadata");
        assert_eq!(metadata.name, CLASSIFIER);
        assert_eq!(metadata.inputs[0].shape, vec![-1, 8]);

        assert_eq!(
            metadata.outputs.len(),
            2,
            "{transport}: a band and its odds"
        );

        let band = &metadata.outputs[0];
        assert_eq!(band.name, "band");
        assert_eq!(
            band.datatype,
            DataType::Int64,
            "{transport}: the label is an integer class index"
        );
        assert_eq!(band.shape, vec![-1]);

        let probabilities = &metadata.outputs[1];
        assert_eq!(probabilities.name, "probabilities");
        assert_eq!(probabilities.datatype, DataType::Fp32);
        assert_eq!(
            probabilities.shape,
            vec![-1, 3],
            "{transport}: one column per class"
        );
    }
}

/// A regression prediction, over both transports, for a batch of two rows.
#[tokio::test]
async fn the_regression_model_predicts_energy_use_per_row() {
    let server = server().await;

    for transport in [Transport::Http, Transport::Grpc] {
        let client = server.client(transport).await;
        let response = client
            .infer(request(REGRESSION, &[BIG_OFFICE, SMALL_FLAT]))
            .await
            .expect("the prediction succeeds");

        assert_eq!(response.model_name, REGRESSION);
        assert_eq!(response.model_version, "1");
        assert_eq!(response.outputs.len(), 1);

        let kwh = &response.outputs[0];
        assert_eq!(kwh.name, "kwh");
        assert_eq!(
            kwh.shape,
            vec![2, 1],
            "{transport}: one prediction per row sent"
        );

        let values = fp32(kwh);
        assert_eq!(values.len(), 2);
        for (row, value) in values.iter().enumerate() {
            assert!(
                value.is_finite() && *value > 0.0,
                "{transport}: row {row} predicts a positive kWh, got {value}"
            );
        }
        assert!(
            values[0] > values[1],
            "{transport}: the large cold office uses more than the small warm flat, \
             got {:?}",
            values
        );
    }
}

/// A classification, over both transports. The label must agree with the
/// probabilities, which must be a distribution.
#[tokio::test]
async fn the_classifier_returns_a_band_that_agrees_with_its_probabilities() {
    let server = server().await;
    let rows = [BIG_OFFICE, SMALL_FLAT];

    for transport in [Transport::Http, Transport::Grpc] {
        let client = server.client(transport).await;
        let response = client
            .infer(request(CLASSIFIER, &rows))
            .await
            .expect("the classification succeeds");

        assert_eq!(
            response.outputs.len(),
            2,
            "{transport}: a band and its odds"
        );

        let band = response
            .outputs
            .iter()
            .find(|t| t.name == "band")
            .expect("a band output");
        let bands = int64(band);
        assert_eq!(bands.len(), rows.len());
        for (row, class) in bands.iter().enumerate() {
            assert!(
                (0..3).contains(class),
                "{transport}: row {row} names one of three classes, got {class}"
            );
        }

        let probabilities = response
            .outputs
            .iter()
            .find(|t| t.name == "probabilities")
            .expect("a probabilities output");
        assert_eq!(probabilities.shape, vec![rows.len() as i64, 3]);

        let odds = fp32(probabilities);
        for (row, chunk) in odds.chunks(3).enumerate() {
            let total: f32 = chunk.iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-4,
                "{transport}: row {row} probabilities sum to one, got {total}"
            );

            // The label must be the most likely class, or the two outputs
            // disagree about the same row.
            let most_likely = chunk
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(index, _)| index as i64)
                .expect("three classes");
            assert_eq!(
                bands[row], most_likely,
                "{transport}: row {row} band matches its most likely class"
            );
        }
    }
}

/// Batching must not change the answer: the same row predicted alone and
/// inside a batch must come back the same, or a busy server quietly returns
/// different numbers from an idle one.
#[tokio::test]
async fn a_row_predicts_the_same_alone_as_it_does_in_a_batch() {
    let server = server().await;
    let client = server.client(Transport::Http).await;

    let alone = client
        .infer(request(REGRESSION, &[BIG_OFFICE]))
        .await
        .expect("the single-row prediction succeeds");
    let batched = client
        .infer(request(
            REGRESSION,
            &[SMALL_FLAT, BIG_OFFICE, SMALL_FLAT, BIG_OFFICE],
        ))
        .await
        .expect("the batched prediction succeeds");

    let alone = fp32(&alone.outputs[0]);
    let batched = fp32(&batched.outputs[0]);

    assert_eq!(batched.len(), 4);
    for (row, value) in [(1, batched[1]), (3, batched[3])] {
        assert!(
            (value - alone[0]).abs() < 1e-4,
            "row {row} of the batch matches the same row predicted alone: \
             {value} against {}",
            alone[0]
        );
    }
}

/// A request for a model that is not there must be refused as such, not as a
/// server error, and must say so the same way on both transports.
#[tokio::test]
async fn a_model_that_is_not_served_is_refused_as_missing() {
    let server = server().await;

    for transport in [Transport::Http, Transport::Grpc] {
        let client = server.client(transport).await;
        let missing = ModelRef::new("no_such_model".to_string(), None);

        assert!(
            !client.model_ready(&missing).await.unwrap_or(false),
            "{transport}: an absent model is not ready"
        );

        let error = client
            .infer(request("no_such_model", &[BIG_OFFICE]))
            .await
            .expect_err("inference on an absent model fails");
        let message = error.to_string().to_lowercase();
        assert!(
            message.contains("not found") || message.contains("no_such_model"),
            "{transport}: the failure names the missing model, got {error}"
        );
    }
}

/// A row of the wrong width must be refused by the server, whatever the
/// client's own checks do.
#[tokio::test]
async fn a_tensor_of_the_wrong_shape_is_refused() {
    let server = server().await;
    let client = server.client(Transport::Http).await;

    let mut bad = request(REGRESSION, &[BIG_OFFICE]);
    // Seven features where the model takes eight.
    bad.inputs = vec![Tensor::new(
        "features",
        vec![1, 7],
        TensorData::Fp32(vec![1.0; 7]),
    )];

    let error = client
        .infer(bad)
        .await
        .expect_err("a mis-shaped tensor is refused");
    let message = error.to_string().to_lowercase();
    assert!(
        message.contains("shape") || message.contains("invalid") || message.contains("8"),
        "the failure explains the shape, got {error}"
    );
}

/// The Prometheus endpoint: it must be served on the REST port, parse as
/// Prometheus text, carry the metrics the load tool reads, and move when
/// inferences are run.
#[tokio::test]
async fn the_prometheus_endpoint_counts_the_inferences_it_served() {
    let server = server().await;
    let client = server.client(Transport::Http).await;

    let before = scrape(&server.rest).await.expect("the metrics parse");
    let counted = |scrape: &galemind_client::Scrape, model: &str| {
        scrape
            .value_where("gm_inference_count", &[("model", model)])
            .unwrap_or(0.0)
    };

    // The metric must exist before anything is asked of it, so that a
    // dashboard does not start with a gap.
    assert!(
        before.value("gm_cpu_utilization").is_some(),
        "cpu utilisation is reported"
    );
    assert!(
        before.value("gm_memory_usage_bytes").is_some(),
        "memory use is reported"
    );

    let runs = 5;
    for _ in 0..runs {
        client
            .infer(request(REGRESSION, &[BIG_OFFICE, SMALL_FLAT]))
            .await
            .expect("the prediction succeeds");
    }

    let after = scrape(&server.rest).await.expect("the metrics parse");
    let served = counted(&after, REGRESSION) - counted(&before, REGRESSION);
    assert_eq!(
        served, runs as f64,
        "the inference counter moved by the requests served"
    );

    // The histogram the dashboards read for latency must have observations.
    assert!(
        after
            .value_where(
                "gm_inference_request_duration_seconds_count",
                &[("model", REGRESSION), ("version", "1")],
            )
            .unwrap_or(0.0)
            >= runs as f64,
        "inference latency is observed, not just counted"
    );
}

/// The endpoint must be plain Prometheus text on the documented path, since
/// that is what a scraper configuration points at.
#[tokio::test]
async fn the_metrics_endpoint_serves_prometheus_text_at_slash_metrics() {
    let server = server().await;

    let response = reqwest::get(format!("{}/metrics", server.rest))
        .await
        .expect("/metrics answers");
    assert!(
        response.status().is_success(),
        "/metrics returns success, got {}",
        response.status()
    );

    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(
        content_type.starts_with("text/plain"),
        "/metrics is served as Prometheus text, got {content_type:?}"
    );

    let body = response.text().await.expect("a body");
    assert!(
        body.contains("# HELP gm_inference_count"),
        "the exposition carries HELP lines"
    );
    assert!(
        body.contains("# TYPE gm_inference_count"),
        "the exposition carries TYPE lines"
    );
}

// -- the catalogue endpoints, over the transport that serves them -----------

/// The catalogue is an HTTP extension, so it answers there and says so
/// elsewhere rather than listing over a different port than the caller chose.
#[tokio::test]
async fn the_catalogue_lists_the_repository_over_http_and_is_refused_over_grpc() {
    let server = server().await;

    let catalogue = server
        .client(Transport::Http)
        .await
        .model_list()
        .await
        .expect("the catalogue answers over HTTP");

    assert_eq!(catalogue.count, catalogue.models.len());
    assert!(
        catalogue.count >= 10,
        "the image carries the whole zoo: {:?}",
        catalogue.models.iter().map(|m| &m.name).collect::<Vec<_>>()
    );

    let regression = catalogue
        .models
        .iter()
        .find(|model| model.name == REGRESSION)
        .expect("the regression model is listed");
    assert_eq!(regression.platform, "onnx_onnxv1");
    assert_eq!(regression.latest_version, "1");
    // The image runs with --preload, so everything is in memory already.
    assert!(
        regression.ready,
        "a preloaded model is ready without a load"
    );

    let error = server
        .client(Transport::Grpc)
        .await
        .model_list()
        .await
        .expect_err("gRPC has no catalogue call");
    assert!(error.to_string().contains("/v2/models/list"), "{error}");
}

/// Every version of a model, with the signature each one reports, and which
/// version an unversioned request would resolve to.
#[tokio::test]
async fn model_info_reports_every_version_of_a_model() {
    let server = server().await;
    let client = server.client(Transport::Http).await;

    let info = client
        .model_info(REGRESSION, true)
        .await
        .expect("info answers");
    assert_eq!(info.name, REGRESSION);
    assert_eq!(info.latest_version, "1");

    let version = info.versions.first().expect("one version");
    assert_eq!(version.version, "1");
    assert!(version.is_latest);
    assert!(version.loaded, "the image preloads every model");
    assert_eq!(version.inputs[0].name, "features");
    assert_eq!(version.inputs[0].datatype, "FP32");
    assert_eq!(version.outputs[0].name, "kwh");
    // The image serves from RocksDB, which records the artifact's digest, so
    // two deployments can be compared by it.
    assert!(!version.digest.is_empty(), "{version:?}");

    // The model with two versions is the one that shows resolution working.
    let info = client
        .model_info("occupancy_mlp", false)
        .await
        .expect("info answers");
    assert_eq!(info.latest_version, "2");
    assert_eq!(
        info.versions
            .iter()
            .map(|v| v.version.as_str())
            .collect::<Vec<_>>(),
        vec!["1", "2"]
    );

    let error = client
        .model_info("ghost", false)
        .await
        .expect_err("an unknown model is an error");
    assert!(error.is_client_error(), "{error}");
}
