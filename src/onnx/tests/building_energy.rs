//! Runs the committed building-energy models through the ONNX backend.

use galemind_core::{
    Backend, BackendFactory, DataType, InferRequest, ModelArtifact, ModelConfig, ModelKey,
    ModelRef, Parameters, Tensor, TensorData, PLATFORM_ONNX,
};
use galemind_onnx::OnnxBackendFactory;
use std::path::PathBuf;
use std::sync::Arc;

/// Two buildings: a large poorly insulated office on a cold evening, and a
/// small well insulated flat on a mild sunny afternoon.
const ROWS: [[f32; 8]; 2] = [
    [1800.0, 30.0, -4.0, 18.0, 22.0, 1.0, 0.0, 40.0],
    [55.0, 2.0, 18.0, 14.0, 21.0, 5.0, 1.0, 600.0],
];

fn repository() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")
}

fn artifact(name: &str) -> ModelArtifact {
    let path = repository().join(name).join("1/model.onnx");
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("run `make models` first: cannot read {path:?}: {e}"));
    ModelArtifact {
        key: ModelKey::new(name, "1"),
        platform: PLATFORM_ONNX.to_string(),
        digest: "test".to_string(),
        config: ModelConfig {
            platform: None,
            max_batch_size: Some(32),
            intra_op_threads: Some(1),
        },
        bytes,
    }
}

fn load(name: &str) -> Arc<dyn Backend> {
    OnnxBackendFactory::new(1).load(&artifact(name)).unwrap()
}

fn request(name: &str, rows: &[[f32; 8]]) -> InferRequest {
    let data: Vec<f32> = rows.iter().flatten().copied().collect();
    InferRequest {
        model: ModelRef::latest(name),
        id: Some("test-1".to_string()),
        parameters: Parameters::new(),
        inputs: vec![Tensor::new(
            "features",
            vec![rows.len() as i64, 8],
            TensorData::Fp32(data),
        )],
        requested_outputs: Vec::new(),
    }
}

#[test]
fn metadata_is_read_from_the_artifact_not_hardcoded() {
    let backend = load("building_energy");
    let metadata = backend.metadata();

    assert_eq!(metadata.platform, PLATFORM_ONNX);
    assert_eq!(metadata.inputs.len(), 1);
    assert_eq!(metadata.inputs[0].name, "features");
    assert_eq!(metadata.inputs[0].datatype, DataType::Fp32);
    assert_eq!(metadata.inputs[0].shape, vec![-1, 8]);
    assert_eq!(metadata.outputs[0].name, "kwh");
    assert_eq!(metadata.outputs[0].shape, vec![-1, 1]);
    assert!(
        metadata.is_batchable(),
        "a dynamic leading dimension makes the model batchable"
    );
    assert!(backend.size_bytes() > 1_000);
}

#[test]
fn the_regressor_predicts_more_energy_for_the_larger_colder_building() {
    let backend = load("building_energy");
    let response = backend.infer(&request("building_energy", &ROWS)).unwrap();

    assert_eq!(response.model_name, "building_energy");
    assert_eq!(response.model_version, "1");
    assert_eq!(response.id.as_deref(), Some("test-1"));
    assert_eq!(response.outputs.len(), 1);

    let output = &response.outputs[0];
    assert_eq!(output.name, "kwh");
    assert_eq!(output.shape, vec![2, 1]);
    let TensorData::Fp32(kwh) = &output.data else {
        panic!("expected FP32 output, got {}", output.datatype());
    };
    assert!(
        kwh[0] > kwh[1],
        "large cold office should use more: {kwh:?}"
    );
    assert!(kwh.iter().all(|v| *v > 0.0), "{kwh:?}");
}

#[test]
fn the_classifier_returns_a_band_and_a_probability_row_per_input() {
    let backend = load("building_energy_class");
    let response = backend
        .infer(&request("building_energy_class", &ROWS))
        .unwrap();

    let band = &response.outputs[0];
    assert_eq!(band.name, "band");
    assert_eq!(band.shape, vec![2]);
    let TensorData::Int64(bands) = &band.data else {
        panic!("expected INT64 band, got {}", band.datatype());
    };
    assert!(bands.iter().all(|b| (0..3).contains(b)), "{bands:?}");

    let probabilities = &response.outputs[1];
    assert_eq!(probabilities.shape, vec![2, 3]);
    let TensorData::Fp32(values) = &probabilities.data else {
        panic!("expected FP32 probabilities");
    };
    for row in values.chunks(3) {
        let total: f32 = row.iter().sum();
        assert!((total - 1.0).abs() < 1e-4, "row does not sum to 1: {row:?}");
    }
}

#[test]
fn a_batch_of_rows_gives_the_same_answer_as_running_each_row_alone() {
    let backend = load("building_energy");

    let batched = backend.infer(&request("building_energy", &ROWS)).unwrap();
    let TensorData::Fp32(batched) = &batched.outputs[0].data else {
        panic!("expected FP32 output");
    };

    for (index, row) in ROWS.iter().enumerate() {
        let single = backend.infer(&request("building_energy", &[*row])).unwrap();
        let TensorData::Fp32(single) = &single.outputs[0].data else {
            panic!("expected FP32 output");
        };
        assert_eq!(single.len(), 1);
        assert!(
            (single[0] - batched[index]).abs() < 1e-3,
            "row {index}: batched {} vs single {}",
            batched[index],
            single[0]
        );
    }
}

#[test]
fn a_wrongly_shaped_input_is_refused_by_the_runtime() {
    let backend = load("building_energy");
    let bad = InferRequest {
        inputs: vec![Tensor::new(
            "features",
            vec![1, 3],
            TensorData::Fp32(vec![1.0, 2.0, 3.0]),
        )],
        ..request("building_energy", &ROWS)
    };
    assert!(backend.infer(&bad).is_err());
}
