//! Checks the HTTP surface against the Predict Protocol v2 specification,
//! driving the real service over the committed ONNX models.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use galemind_core::{BackendRegistry, Inference, MetricsSource};
use galemind_onnx::OnnxBackendFactory;
use galemind_service::ServiceBuilder;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use tower::ServiceExt;

/// A large, poorly insulated office on a cold evening.
const OFFICE: [f32; 8] = [1800.0, 30.0, -4.0, 18.0, 22.0, 1.0, 0.0, 40.0];

async fn app() -> Router {
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models");
    let backends = BackendRegistry::new().with(Arc::new(OnnxBackendFactory::new(1)));
    let service = ServiceBuilder::new(backends)
        .with_filesystem_store(repository)
        .build()
        .expect("service builds");
    service.start(false).await.expect("repository is readable");
    rest_server::app(
        Arc::clone(&service) as Arc<dyn Inference>,
        Some(service as Arc<dyn MetricsSource>),
    )
}

async fn get(path: &str) -> (StatusCode, Vec<u8>) {
    send(Request::get(path).body(Body::empty()).unwrap()).await
}

async fn post(path: &str, body: Value) -> (StatusCode, Vec<u8>) {
    send(
        Request::post(path)
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap(),
    )
    .await
}

async fn send(request: Request<Body>) -> (StatusCode, Vec<u8>) {
    let response = app().await.oneshot(request).await.unwrap();
    let status = response.status();
    let body = response.into_body().collect().await.unwrap().to_bytes();
    (status, body.to_vec())
}

fn parse(body: &[u8]) -> Value {
    serde_json::from_slice(body).expect("a JSON body")
}

fn infer_body(rows: usize) -> Value {
    let data: Vec<f32> = (0..rows).flat_map(|_| OFFICE).collect();
    json!({
        "id": "req-42",
        "inputs": [{
            "name": "features",
            "shape": [rows, 8],
            "datatype": "FP32",
            "data": data
        }]
    })
}

#[tokio::test]
async fn the_liveness_endpoint_answers_200_with_an_empty_body() {
    let (status, body) = get("/v2/health/live").await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.is_empty(), "the specification requires an empty body");
}

#[tokio::test]
async fn the_readiness_endpoints_answer_200_with_an_empty_body() {
    for path in [
        "/v2/health/ready",
        "/v2/models/building_energy/ready",
        "/v2/models/building_energy/versions/1/ready",
    ] {
        let (status, body) = get(path).await;
        assert_eq!(status, StatusCode::OK, "{path}");
        assert!(body.is_empty(), "{path}");
    }
}

#[tokio::test]
async fn readiness_of_an_unknown_model_is_reported_as_false_with_a_4xx() {
    for path in [
        "/v2/models/ghost/ready",
        "/v2/models/building_energy/versions/9/ready",
    ] {
        let (status, _) = get(path).await;
        assert!(status.is_client_error(), "{path} answered {status}");
    }
}

#[tokio::test]
async fn server_metadata_reports_the_name_version_and_extensions() {
    let (status, body) = get("/v2").await;
    assert_eq!(status, StatusCode::OK);

    let metadata = parse(&body);
    assert_eq!(metadata["name"], "galemind");
    assert!(metadata["version"].is_string());
    assert_eq!(metadata["extensions"], json!([]));
}

#[tokio::test]
async fn model_metadata_uses_the_specifications_field_names() {
    let (status, body) = get("/v2/models/building_energy").await;
    assert_eq!(status, StatusCode::OK);

    let metadata = parse(&body);
    assert_eq!(metadata["name"], "building_energy");
    assert_eq!(metadata["versions"], json!(["1"]));
    assert_eq!(metadata["platform"], "onnx_onnxv1");
    assert_eq!(metadata["inputs"][0]["name"], "features");
    assert_eq!(metadata["inputs"][0]["datatype"], "FP32");
    assert_eq!(metadata["inputs"][0]["shape"], json!([-1, 8]));
    assert_eq!(metadata["outputs"][0]["name"], "kwh");
}

#[tokio::test]
async fn model_metadata_is_also_served_per_version() {
    let (status, body) = get("/v2/models/building_energy/versions/1").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(parse(&body)["name"], "building_energy");
}

#[tokio::test]
async fn metadata_for_an_unknown_model_is_an_error_object() {
    let (status, body) = get("/v2/models/ghost").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(
        parse(&body)["error"].as_str().unwrap().contains("ghost"),
        "{:?}",
        parse(&body)
    );
}

#[tokio::test]
async fn an_inference_response_carries_the_specifications_fields() {
    let (status, body) = post("/v2/models/building_energy/infer", infer_body(2)).await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));

    let response = parse(&body);
    assert_eq!(response["model_name"], "building_energy");
    assert_eq!(response["model_version"], "1");
    assert_eq!(response["id"], "req-42", "the request id must be returned");

    let output = &response["outputs"][0];
    assert_eq!(output["name"], "kwh");
    assert_eq!(output["datatype"], "FP32");
    assert_eq!(output["shape"], json!([2, 1]));
    assert_eq!(output["data"].as_array().unwrap().len(), 2);
    assert!(output["data"][0].as_f64().unwrap() > 0.0);
    assert!(
        output.get("parameters").is_none(),
        "no server-specific parameters belong in an ordinary response"
    );
}

#[tokio::test]
async fn inference_is_also_served_per_version() {
    let (status, body) = post("/v2/models/building_energy/versions/1/infer", infer_body(1)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(parse(&body)["model_version"], "1");
}

#[tokio::test]
async fn the_classifier_returns_both_of_its_outputs() {
    let (status, body) = post("/v2/models/building_energy_class/infer", infer_body(3)).await;
    assert_eq!(status, StatusCode::OK);

    let response = parse(&body);
    let outputs = response["outputs"].as_array().unwrap();
    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0]["name"], "band");
    assert_eq!(outputs[0]["datatype"], "INT64");
    assert_eq!(outputs[0]["shape"], json!([3]));
    assert_eq!(outputs[1]["name"], "probabilities");
    assert_eq!(outputs[1]["shape"], json!([3, 3]));
}

#[tokio::test]
async fn only_the_requested_outputs_are_returned() {
    let mut body = infer_body(1);
    body["outputs"] = json!([{"name": "probabilities"}]);

    let (status, body) = post("/v2/models/building_energy_class/infer", body).await;
    assert_eq!(status, StatusCode::OK);

    let outputs = parse(&body)["outputs"].as_array().unwrap().clone();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0]["name"], "probabilities");
}

#[tokio::test]
async fn nested_tensor_data_is_accepted_as_the_specification_allows() {
    let body = json!({
        "inputs": [{
            "name": "features",
            "shape": [1, 8],
            "datatype": "FP32",
            "data": [[1800.0, 30.0, -4.0, 18.0, 22.0, 1.0, 0.0, 40.0]]
        }]
    });
    let (status, body) = post("/v2/models/building_energy/infer", body).await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
}

#[tokio::test]
async fn a_request_without_an_id_gets_a_response_without_one() {
    let mut body = infer_body(1);
    body.as_object_mut().unwrap().remove("id");

    let (status, body) = post("/v2/models/building_energy/infer", body).await;
    assert_eq!(status, StatusCode::OK);
    assert!(parse(&body).get("id").is_none());
}

#[tokio::test]
async fn failures_are_reported_as_the_specifications_error_object() {
    let cases = vec![
        (
            "an unknown model",
            "/v2/models/ghost/infer",
            infer_body(1),
            StatusCode::NOT_FOUND,
        ),
        (
            "an unknown version",
            "/v2/models/building_energy/versions/7/infer",
            infer_body(1),
            StatusCode::NOT_FOUND,
        ),
        (
            "an unknown datatype",
            "/v2/models/building_energy/infer",
            json!({"inputs": [
                {"name": "features", "shape": [1, 8], "datatype": "MAGIC", "data": [0,0,0,0,0,0,0,0]}
            ]}),
            StatusCode::BAD_REQUEST,
        ),
        (
            "a shape the model does not accept",
            "/v2/models/building_energy/infer",
            json!({"inputs": [
                {"name": "features", "shape": [1, 3], "datatype": "FP32", "data": [1.0, 2.0, 3.0]}
            ]}),
            StatusCode::BAD_REQUEST,
        ),
        (
            "an unknown input name",
            "/v2/models/building_energy/infer",
            json!({"inputs": [
                {"name": "wrong", "shape": [1, 8], "datatype": "FP32", "data": [0,0,0,0,0,0,0,0]}
            ]}),
            StatusCode::BAD_REQUEST,
        ),
        (
            "no inputs at all",
            "/v2/models/building_energy/infer",
            json!({"inputs": []}),
            StatusCode::BAD_REQUEST,
        ),
    ];

    for (what, path, body, expected) in cases {
        let (status, body) = post(path, body).await;
        assert_eq!(status, expected, "{what}");
        assert!(
            parse(&body)["error"].is_string(),
            "{what}: body must be an error object, got {}",
            String::from_utf8_lossy(&body)
        );
    }
}

#[tokio::test]
async fn a_malformed_body_is_reported_as_an_error_object_not_plain_text() {
    let response = app()
        .await
        .oneshot(
            Request::post("/v2/models/building_energy/infer")
                .header("content-type", "application/json")
                .body(Body::from("{ not json"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    assert!(parse(&body)["error"].is_string(), "{body:?}");
}

#[tokio::test]
async fn the_protocol_version_in_the_url_is_literal() {
    // A captured path segment would have answered any version at all.
    for path in [
        "/v3/health/live",
        "/v1/health/live",
        "/anything/health/live",
    ] {
        let (status, body) = get(path).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{path}");
        assert!(parse(&body)["error"].is_string(), "{path}");
    }
}

#[tokio::test]
async fn endpoints_removed_with_the_chat_layer_are_gone() {
    for path in ["/v1/chat/completions", "/v1/models", "/v2/models"] {
        let (status, _) = get(path).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{path} should not exist");
    }
}

// -- metrics, which are not part of the protocol --------------------------

#[tokio::test]
async fn metrics_are_served_as_prometheus_text_exposition() {
    let response = app()
        .await
        .oneshot(Request::get("/metrics").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("text/plain; version=0.0.4; charset=utf-8")
    );

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8(body.to_vec()).unwrap();
    for name in [
        "gm_server_info",
        "gm_inference_request_success",
        "gm_inference_count",
        "gm_inference_exec_count",
        "gm_model_cache_entries",
        "gm_cpu_utilization",
        "gm_memory_usage_bytes",
    ] {
        assert!(text.contains(&format!("# TYPE {name} ")), "missing {name}");
    }
}

#[tokio::test]
async fn an_inference_is_counted_in_the_metrics() {
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models");
    let backends = BackendRegistry::new().with(Arc::new(OnnxBackendFactory::new(1)));
    let service = ServiceBuilder::new(backends)
        .with_filesystem_store(repository)
        .build()
        .unwrap();
    service.start(false).await.unwrap();

    let app = rest_server::app(
        Arc::clone(&service) as Arc<dyn Inference>,
        Some(Arc::clone(&service) as Arc<dyn MetricsSource>),
    );

    let response = app
        .clone()
        .oneshot(
            Request::post("/v2/models/building_energy/infer")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&infer_body(4)).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let scraped = app
        .oneshot(Request::get("/metrics").body(Body::empty()).unwrap())
        .await
        .unwrap();
    let body = scraped.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8(body.to_vec()).unwrap();

    assert!(
        text.contains("gm_inference_request_success{model=\"building_energy\",version=\"1\"} 1"),
        "{text}"
    );
    // A four-row request is one request but four inferences.
    assert!(
        text.contains("gm_inference_count{model=\"building_energy\",version=\"1\"} 4"),
        "{text}"
    );
    assert!(
        text.contains("gm_inference_exec_count{model=\"building_energy\",version=\"1\"} 1"),
        "{text}"
    );
}

#[tokio::test]
async fn a_failed_request_is_counted_with_its_reason() {
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models");
    let backends = BackendRegistry::new().with(Arc::new(OnnxBackendFactory::new(1)));
    let service = ServiceBuilder::new(backends)
        .with_filesystem_store(repository)
        .build()
        .unwrap();
    service.start(false).await.unwrap();

    let app = rest_server::app(
        Arc::clone(&service) as Arc<dyn Inference>,
        Some(Arc::clone(&service) as Arc<dyn MetricsSource>),
    );

    let bad = json!({"inputs": [
        {"name": "features", "shape": [1, 3], "datatype": "FP32", "data": [1.0, 2.0, 3.0]}
    ]});
    let response = app
        .clone()
        .oneshot(
            Request::post("/v2/models/building_energy/infer")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&bad).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let scraped = app
        .oneshot(Request::get("/metrics").body(Body::empty()).unwrap())
        .await
        .unwrap();
    let body = scraped.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8(body.to_vec()).unwrap();

    assert!(
        text.contains(
            "gm_inference_request_failure{model=\"building_energy\",version=\"1\",\
             reason=\"invalid\"} 1"
        ),
        "{text}"
    );
}

#[tokio::test]
async fn a_server_built_without_metrics_does_not_serve_them() {
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models");
    let backends = BackendRegistry::new().with(Arc::new(OnnxBackendFactory::new(1)));
    let service = ServiceBuilder::new(backends)
        .with_filesystem_store(repository)
        .build()
        .unwrap();
    service.start(false).await.unwrap();

    let response = rest_server::app(service as Arc<dyn Inference>, None)
        .oneshot(Request::get("/metrics").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

// -- the catalogue endpoints, which the protocol does not define ------------

#[tokio::test]
async fn the_catalogue_lists_every_model_with_its_versions_and_newest() {
    let (status, body) = get("/v2/models/list").await;
    assert_eq!(status, StatusCode::OK);

    let body = parse(&body);
    let models = body["models"].as_array().expect("an array of models");
    assert_eq!(
        body["count"].as_u64(),
        Some(models.len() as u64),
        "the count matches the array"
    );
    assert!(
        models.len() >= 2,
        "the committed repository holds at least the two energy models: {body}"
    );

    let energy = models
        .iter()
        .find(|model| model["name"] == "building_energy")
        .expect("building_energy is listed");
    assert_eq!(energy["platform"], "onnx_onnxv1");
    assert_eq!(energy["versions"], json!(["1"]));
    assert_eq!(energy["latest_version"], "1");
    // Nothing was inferred, so nothing is loaded: listing must not load.
    assert_eq!(energy["loaded_versions"], json!([]));
    assert_eq!(energy["ready"], json!(false));
}

#[tokio::test]
async fn model_info_reports_every_version_and_its_signature() {
    let (status, body) = get("/v2/models/info/building_energy").await;
    assert_eq!(status, StatusCode::OK);

    let body = parse(&body);
    assert_eq!(body["name"], "building_energy");
    assert_eq!(body["latest_version"], "1");

    let versions = body["versions"].as_array().expect("an array of versions");
    assert_eq!(versions.len(), 1, "{body}");
    let version = &versions[0];
    assert_eq!(version["version"], "1");
    assert_eq!(version["platform"], "onnx_onnxv1");
    assert_eq!(version["is_latest"], json!(true));
    // Loading is the default, so the signature is reported.
    assert_eq!(version["loaded"], json!(true));
    assert_eq!(version["inputs"][0]["name"], "features");
    assert_eq!(version["inputs"][0]["datatype"], "FP32");
    assert_eq!(version["inputs"][0]["shape"], json!([-1, 8]));
    assert_eq!(version["outputs"][0]["name"], "kwh");
    assert!(
        version["size_bytes"].as_u64().unwrap_or(0) > 0,
        "the artifact's size is known: {version}"
    );
}

/// A caller that only wants the versions and their settings should not have to
/// pay for a load of a model larger than the cache.
#[tokio::test]
async fn model_info_can_be_asked_not_to_load_anything() {
    let (status, body) = get("/v2/models/info/building_energy?load=false").await;
    assert_eq!(status, StatusCode::OK);

    let version = &parse(&body)["versions"][0];
    assert_eq!(version["version"], "1");
    assert_eq!(version["loaded"], json!(false));
    assert_eq!(
        version["inputs"],
        json!([]),
        "a signature comes from a loaded session, and nothing was loaded"
    );
}

#[tokio::test]
async fn model_info_for_an_unknown_model_is_a_protocol_error() {
    let (status, body) = get("/v2/models/info/ghost").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(parse(&body)["error"].is_string(), "{body:?}");
}

/// `list` and `info` are static segments, so they win over the metadata URL's
/// parameter. The protocol's own URLs must be unaffected by that.
#[tokio::test]
async fn the_catalogue_urls_do_not_shadow_the_protocols_model_urls() {
    let (status, body) = get("/v2/models/building_energy").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(parse(&body)["name"], "building_energy");

    let (status, _) = get("/v2/models/building_energy/ready").await;
    assert_eq!(status, StatusCode::OK);
}

// -- per-request tracing ---------------------------------------------------

#[tokio::test]
async fn every_response_carries_a_request_id() {
    let response = app()
        .await
        .oneshot(Request::get("/v2/health/live").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let id = response
        .headers()
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .expect("an id is echoed");
    assert!(!id.is_empty());
}

#[tokio::test]
async fn a_callers_request_id_is_echoed_back() {
    let response = app()
        .await
        .oneshot(
            Request::get("/v2/health/live")
                .header("x-request-id", "gateway-99")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response
            .headers()
            .get("x-request-id")
            .and_then(|value| value.to_str().ok()),
        Some("gateway-99")
    );
}

/// A request that never reaches a handler is still traced, which is the case
/// that matters: an unknown URL is what a misconfigured client sends.
#[tokio::test]
async fn a_request_for_an_unknown_url_is_traced_too() {
    let response = app()
        .await
        .oneshot(Request::get("/v2/nope").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    assert!(response.headers().contains_key("x-request-id"));
}
