//! The service end to end: real ONNX models, real cache, real batching.

use galemind_core::{
    BackendRegistry, DataType, Inference, InferError, InferRequest, ModelKey, ModelRef,
    Parameters, Tensor, TensorData, PLATFORM_ONNX,
};
use galemind_onnx::OnnxBackendFactory;
use galemind_service::ServiceBuilder;
use std::path::PathBuf;
use std::sync::Arc;

fn repository() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")
}

async fn service() -> Arc<galemind_service::InferenceService> {
    let backends = BackendRegistry::new().with(Arc::new(OnnxBackendFactory::new(1)));
    let service = ServiceBuilder::new(backends)
        .with_filesystem_store(repository())
        .build()
        .expect("service builds");
    service.start(false).await.expect("repository is readable");
    service
}

/// A large, poorly insulated office on a cold evening.
const OFFICE: [f32; 8] = [1800.0, 30.0, -4.0, 18.0, 22.0, 1.0, 0.0, 40.0];
/// A small, well insulated flat on a mild sunny afternoon.
const FLAT: [f32; 8] = [55.0, 2.0, 18.0, 14.0, 21.0, 5.0, 1.0, 600.0];

fn request(model: &str, rows: &[[f32; 8]]) -> InferRequest {
    InferRequest {
        model: ModelRef::latest(model),
        id: Some(format!("{model}-request")),
        parameters: Parameters::new(),
        inputs: vec![Tensor::new(
            "features",
            vec![rows.len() as i64, 8],
            TensorData::Fp32(rows.iter().flatten().copied().collect()),
        )],
        requested_outputs: Vec::new(),
    }
}

#[tokio::test]
async fn both_models_in_the_repository_are_served() {
    let service = service().await;
    assert_eq!(
        service.models().await.unwrap(),
        vec![
            ModelKey::new("building_energy", "1"),
            ModelKey::new("building_energy_class", "1"),
        ]
    );
    assert!(service.is_ready().await);
    assert!(
        service
            .is_model_ready(&ModelRef::latest("building_energy"))
            .await
    );
    assert!(!service.is_model_ready(&ModelRef::latest("ghost")).await);
}

#[tokio::test]
async fn metadata_combines_the_runtimes_tensors_with_the_repositorys_versions() {
    let service = service().await;
    let metadata = service
        .metadata(&ModelRef::latest("building_energy"))
        .await
        .unwrap();

    assert_eq!(metadata.name, "building_energy");
    assert_eq!(metadata.versions, vec!["1"]);
    assert_eq!(metadata.platform, PLATFORM_ONNX);
    assert_eq!(metadata.inputs[0].name, "features");
    assert_eq!(metadata.inputs[0].datatype, DataType::Fp32);
    assert_eq!(metadata.inputs[0].shape, vec![-1, 8]);
    assert_eq!(metadata.outputs[0].name, "kwh");
    assert_eq!(metadata.max_batch_size, Some(32));
}

#[tokio::test]
async fn an_inference_names_the_version_it_served_and_echoes_the_request_id() {
    let service = service().await;
    let response = service
        .infer(request("building_energy", &[OFFICE, FLAT]))
        .await
        .unwrap();

    assert_eq!(response.model_name, "building_energy");
    assert_eq!(response.model_version, "1");
    assert_eq!(response.id.as_deref(), Some("building_energy-request"));
    assert_eq!(response.outputs[0].shape, vec![2, 1]);

    let TensorData::Fp32(kwh) = &response.outputs[0].data else {
        panic!("expected FP32 output");
    };
    assert!(kwh[0] > kwh[1], "the office should use more energy: {kwh:?}");
}

#[tokio::test]
async fn an_explicit_version_is_served_and_an_unknown_one_refused() {
    let service = service().await;

    let mut exact = request("building_energy", &[FLAT]);
    exact.model = ModelRef::new("building_energy", Some("1".to_string()));
    assert_eq!(service.infer(exact).await.unwrap().model_version, "1");

    let mut missing = request("building_energy", &[FLAT]);
    missing.model = ModelRef::new("building_energy", Some("4".to_string()));
    let Err(err) = service.infer(missing).await else {
        panic!("an unknown version must be refused");
    };
    assert!(matches!(err, InferError::VersionNotFound { .. }), "{err}");
}

#[tokio::test]
async fn an_unknown_model_is_refused_before_any_work_happens() {
    let service = service().await;
    let Err(err) = service.infer(request("ghost", &[FLAT])).await else {
        panic!("an unknown model must be refused");
    };
    assert!(matches!(err, InferError::ModelNotFound(_)), "{err}");
}

#[tokio::test]
async fn a_request_that_does_not_match_the_model_is_refused_with_a_client_error() {
    let service = service().await;
    let mut bad = request("building_energy", &[FLAT]);
    bad.inputs = vec![Tensor::new(
        "features",
        vec![1, 3],
        TensorData::Fp32(vec![1.0, 2.0, 3.0]),
    )];

    let Err(err) = service.infer(bad).await else {
        panic!("a mis-shaped request must be refused");
    };
    assert!(err.is_client_error(), "{err}");
    assert!(err.to_string().contains("axis 1"), "{err}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn requests_for_different_models_are_demultiplexed() {
    let service = service().await;

    let submissions: Vec<_> = (0..24)
        .map(|i| {
            let service = Arc::clone(&service);
            tokio::spawn(async move {
                let model = if i % 2 == 0 {
                    "building_energy"
                } else {
                    "building_energy_class"
                };
                (model, service.infer(request(model, &[OFFICE])).await)
            })
        })
        .collect();

    for submission in submissions {
        let (model, result) = submission.await.unwrap();
        let response = result.expect("every request should succeed");
        assert_eq!(response.model_name, model);
        let names: Vec<_> = response.outputs.iter().map(|o| o.name.as_str()).collect();
        match model {
            "building_energy" => assert_eq!(names, vec!["kwh"]),
            _ => assert_eq!(names, vec!["band", "probabilities"]),
        }
    }

    // One dispatcher per model version, not per request.
    let metrics = service.dispatcher_metrics().await;
    assert_eq!(metrics.len(), 2, "{metrics:?}");
    for (key, snapshot) in &metrics {
        assert_eq!(snapshot.requests, 12, "{key}: {snapshot:?}");
        assert!(snapshot.batches >= 1);
    }

    // Two models loaded, each only once.
    let stats = service.cache_stats();
    assert_eq!(stats.entries, 2);
    assert_eq!(stats.misses, 2, "{stats:?}");
    service.shutdown().await;
}

#[tokio::test]
async fn a_repository_without_models_is_not_ready() {
    let empty = tempfile::tempdir().unwrap();
    let backends = BackendRegistry::new().with(Arc::new(OnnxBackendFactory::new(1)));
    let service = ServiceBuilder::new(backends)
        .with_filesystem_store(empty.path())
        .build()
        .unwrap();
    service.start(false).await.unwrap();

    assert!(!service.is_ready().await);
    assert!(service.models().await.unwrap().is_empty());
}

#[tokio::test]
async fn preloading_puts_every_model_in_the_cache_before_serving() {
    let backends = BackendRegistry::new().with(Arc::new(OnnxBackendFactory::new(1)));
    let service = ServiceBuilder::new(backends)
        .with_filesystem_store(repository())
        .build()
        .unwrap();

    assert_eq!(service.cache_stats().entries, 0);
    service.start(true).await.unwrap();
    assert_eq!(service.cache_stats().entries, 2);
    assert!(service.is_ready().await);
}

#[tokio::test]
async fn a_service_without_a_store_cannot_be_built() {
    let backends = BackendRegistry::new().with(Arc::new(OnnxBackendFactory::new(1)));
    assert!(ServiceBuilder::new(backends).build().is_err());
    assert!(ServiceBuilder::new(BackendRegistry::new())
        .with_filesystem_store("/tmp")
        .build()
        .is_err());
}
