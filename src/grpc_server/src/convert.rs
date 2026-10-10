//! Translation between the Predict Protocol v2 gRPC messages and the domain.

use crate::pb;
use galemind_core::{
    DataType, InferError, InferRequest, InferResponse, ModelMetadata, ModelRef, Parameter,
    Parameters, RequestedOutput, Tensor, TensorData,
};
use std::collections::HashMap;
use std::str::FromStr;
use tonic::{Code, Status};

type Result<T> = std::result::Result<T, InferError>;

/// Maps a domain error onto the gRPC status the protocol expects.
///
/// The protocol says errors are indicated by the returned status, with OK for
/// success and other codes for failure; it does not dictate which code, so
/// these follow the usual gRPC meanings.
pub fn to_status(error: InferError) -> Status {
    let message = error.to_string();
    let code = match &error {
        InferError::ModelNotFound(_) | InferError::VersionNotFound { .. } => Code::NotFound,
        InferError::InvalidRequest(_) => Code::InvalidArgument,
        InferError::Unsupported(_) => Code::Unimplemented,
        InferError::ModelNotReady(_) => Code::FailedPrecondition,
        InferError::Overloaded => Code::ResourceExhausted,
        InferError::Timeout(_) => Code::DeadlineExceeded,
        InferError::Backend(_) | InferError::Store(_) | InferError::Internal(_) => Code::Internal,
        // A failure relayed from somewhere else, for example by a backend that
        // is itself a client of another server.
        InferError::Remote { client_error, .. } => {
            if *client_error {
                Code::InvalidArgument
            } else {
                Code::Internal
            }
        }
    };
    Status::new(code, message)
}

fn parameters_from_proto(parameters: HashMap<String, pb::InferParameter>) -> Parameters {
    parameters
        .into_iter()
        .filter_map(|(name, parameter)| {
            let value = match parameter.parameter_choice? {
                pb::infer_parameter::ParameterChoice::BoolParam(v) => Parameter::Bool(v),
                pb::infer_parameter::ParameterChoice::Int64Param(v) => Parameter::Int64(v),
                pb::infer_parameter::ParameterChoice::StringParam(v) => Parameter::String(v),
            };
            Some((name, value))
        })
        .collect()
}

fn parameters_to_proto(parameters: &Parameters) -> HashMap<String, pb::InferParameter> {
    parameters
        .iter()
        .map(|(name, value)| {
            let choice = match value {
                Parameter::Bool(v) => pb::infer_parameter::ParameterChoice::BoolParam(*v),
                Parameter::Int64(v) => pb::infer_parameter::ParameterChoice::Int64Param(*v),
                Parameter::String(v) => {
                    pb::infer_parameter::ParameterChoice::StringParam(v.clone())
                }
            };
            (
                name.clone(),
                pb::InferParameter {
                    parameter_choice: Some(choice),
                },
            )
        })
        .collect()
}

/// Number of elements a shape describes, rejecting shapes a request may not use.
fn element_count(name: &str, shape: &[i64]) -> Result<usize> {
    let mut count = 1usize;
    for dim in shape {
        let dim = usize::try_from(*dim).map_err(|_| {
            InferError::InvalidRequest(format!(
                "input '{name}' has dimension {dim}; request shapes must be concrete"
            ))
        })?;
        count = count.checked_mul(dim).ok_or_else(|| {
            InferError::InvalidRequest(format!("input '{name}' has a shape that overflows"))
        })?;
    }
    Ok(count)
}

/// Narrows the wider repeated field the protocol uses for small integer types.
fn narrow<T, U>(name: &str, datatype: DataType, values: &[T]) -> Result<Vec<U>>
where
    T: Copy + std::fmt::Display,
    U: TryFrom<T>,
{
    values
        .iter()
        .map(|value| {
            U::try_from(*value).map_err(|_| {
                InferError::InvalidRequest(format!(
                    "input '{name}' holds {value}, which is outside the range of {datatype}"
                ))
            })
        })
        .collect()
}

/// Reads tensor contents from the typed `contents` field.
fn data_from_contents(
    name: &str,
    datatype: DataType,
    contents: &pb::InferTensorContents,
) -> Result<TensorData> {
    Ok(match datatype {
        DataType::Bool => TensorData::Bool(contents.bool_contents.clone()),
        DataType::Int8 => TensorData::Int8(narrow(name, datatype, &contents.int_contents)?),
        DataType::Int16 => TensorData::Int16(narrow(name, datatype, &contents.int_contents)?),
        DataType::Int32 => TensorData::Int32(contents.int_contents.clone()),
        DataType::Int64 => TensorData::Int64(contents.int64_contents.clone()),
        DataType::Uint8 => TensorData::Uint8(narrow(name, datatype, &contents.uint_contents)?),
        DataType::Uint16 => TensorData::Uint16(narrow(name, datatype, &contents.uint_contents)?),
        DataType::Uint32 => TensorData::Uint32(contents.uint_contents.clone()),
        DataType::Uint64 => TensorData::Uint64(contents.uint64_contents.clone()),
        DataType::Fp32 => TensorData::Fp32(contents.fp32_contents.clone()),
        DataType::Fp64 => TensorData::Fp64(contents.fp64_contents.clone()),
        DataType::Bytes => TensorData::Bytes(contents.bytes_contents.clone()),
        // The protocol has no repeated field for 16-bit floats.
        DataType::Fp16 | DataType::Bf16 => {
            return Err(InferError::InvalidRequest(format!(
                "input '{name}' is {datatype}, which must be sent as raw_input_contents"
            )))
        }
    })
}

/// Builds a domain request from a `ModelInferRequest`.
///
/// Input contents arrive either in the typed `contents` field or, for one tensor
/// per entry and in the same order, in `raw_input_contents`. The protocol
/// forbids using both, and this rejects a request that tries.
pub fn request_from_proto(request: pb::ModelInferRequest) -> Result<InferRequest> {
    let raw = &request.raw_input_contents;
    if !raw.is_empty() && raw.len() != request.inputs.len() {
        return Err(InferError::InvalidRequest(format!(
            "{} raw input contents were sent for {} inputs",
            raw.len(),
            request.inputs.len()
        )));
    }

    let mut inputs = Vec::with_capacity(request.inputs.len());
    for (index, input) in request.inputs.iter().enumerate() {
        let datatype = DataType::from_str(&input.datatype)?;
        let count = element_count(&input.name, &input.shape)?;

        let data = match (raw.get(index), &input.contents) {
            (Some(_), Some(contents)) if !is_empty_contents(contents) => {
                return Err(InferError::InvalidRequest(format!(
                    "input '{}' sets both contents and raw_input_contents",
                    input.name
                )))
            }
            (Some(bytes), _) => TensorData::from_raw(datatype, bytes, count)?,
            (None, Some(contents)) => data_from_contents(&input.name, datatype, contents)?,
            (None, None) => {
                return Err(InferError::InvalidRequest(format!(
                    "input '{}' carries no contents",
                    input.name
                )))
            }
        };

        let tensor = Tensor::new(&input.name, input.shape.clone(), data);
        tensor.validate()?;
        inputs.push(tensor);
    }

    Ok(InferRequest {
        model: ModelRef::new(request.model_name, Some(request.model_version)),
        id: (!request.id.is_empty()).then_some(request.id),
        parameters: parameters_from_proto(request.parameters),
        inputs,
        requested_outputs: request
            .outputs
            .into_iter()
            .map(|output| RequestedOutput {
                name: output.name,
                parameters: parameters_from_proto(output.parameters),
            })
            .collect(),
    })
}

fn is_empty_contents(contents: &pb::InferTensorContents) -> bool {
    contents.bool_contents.is_empty()
        && contents.int_contents.is_empty()
        && contents.int64_contents.is_empty()
        && contents.uint_contents.is_empty()
        && contents.uint64_contents.is_empty()
        && contents.fp32_contents.is_empty()
        && contents.fp64_contents.is_empty()
        && contents.bytes_contents.is_empty()
}

/// Builds a `ModelInferResponse`, returning tensor contents in raw form.
///
/// Raw contents are used because they avoid the per-element protobuf
/// allocations the typed fields require, which the protocol notes as the
/// higher-performance choice.
pub fn response_to_proto(response: InferResponse) -> pb::ModelInferResponse {
    let mut outputs = Vec::with_capacity(response.outputs.len());
    let mut raw_output_contents = Vec::with_capacity(response.outputs.len());

    for tensor in response.outputs {
        raw_output_contents.push(tensor.data.to_raw());
        outputs.push(pb::model_infer_response::InferOutputTensor {
            name: tensor.name,
            datatype: tensor.data.datatype().to_string(),
            shape: tensor.shape,
            parameters: HashMap::new(),
            contents: None,
        });
    }

    pb::ModelInferResponse {
        model_name: response.model_name,
        model_version: response.model_version,
        id: response.id.unwrap_or_default(),
        parameters: parameters_to_proto(&response.parameters),
        outputs,
        raw_output_contents,
    }
}

/// Builds a `ModelMetadataResponse` from model metadata.
pub fn metadata_to_proto(metadata: ModelMetadata) -> pb::ModelMetadataResponse {
    fn tensors(
        tensors: Vec<galemind_core::TensorMetadata>,
    ) -> Vec<pb::model_metadata_response::TensorMetadata> {
        tensors
            .into_iter()
            .map(|tensor| pb::model_metadata_response::TensorMetadata {
                name: tensor.name,
                datatype: tensor.datatype.to_string(),
                shape: tensor.shape,
            })
            .collect()
    }

    pb::ModelMetadataResponse {
        name: metadata.name,
        versions: metadata.versions,
        platform: metadata.platform,
        inputs: tensors(metadata.inputs),
        outputs: tensors(metadata.outputs),
    }
}

/// Builds a `ModelInferRequest` from a domain request, for the client side.
///
/// Contents go in `raw_input_contents`, which the protocol notes is the
/// higher-performance representation and is the only one able to carry FP16 and
/// BF16 exactly.
pub fn request_to_proto(request: &InferRequest) -> pb::ModelInferRequest {
    let mut inputs = Vec::with_capacity(request.inputs.len());
    let mut raw_input_contents = Vec::with_capacity(request.inputs.len());

    for tensor in &request.inputs {
        raw_input_contents.push(tensor.data.to_raw());
        inputs.push(pb::model_infer_request::InferInputTensor {
            name: tensor.name.clone(),
            datatype: tensor.data.datatype().to_string(),
            shape: tensor.shape.clone(),
            parameters: HashMap::new(),
            contents: None,
        });
    }

    pb::ModelInferRequest {
        model_name: request.model.name.clone(),
        model_version: request.model.version.clone().unwrap_or_default(),
        id: request.id.clone().unwrap_or_default(),
        parameters: parameters_to_proto(&request.parameters),
        inputs,
        outputs: request
            .requested_outputs
            .iter()
            .map(
                |output| pb::model_infer_request::InferRequestedOutputTensor {
                    name: output.name.clone(),
                    parameters: parameters_to_proto(&output.parameters),
                },
            )
            .collect(),
        raw_input_contents,
    }
}

/// Reads a `ModelInferResponse` into domain types, for the client side.
pub fn response_from_proto(response: pb::ModelInferResponse) -> Result<InferResponse> {
    let raw = &response.raw_output_contents;
    if !raw.is_empty() && raw.len() != response.outputs.len() {
        return Err(InferError::Internal(format!(
            "the server sent {} raw output contents for {} outputs",
            raw.len(),
            response.outputs.len()
        )));
    }

    let mut outputs = Vec::with_capacity(response.outputs.len());
    for (index, output) in response.outputs.iter().enumerate() {
        let datatype = DataType::from_str(&output.datatype)?;
        let count = element_count(&output.name, &output.shape)?;

        let data = match (raw.get(index), &output.contents) {
            (Some(bytes), _) => TensorData::from_raw(datatype, bytes, count)?,
            (None, Some(contents)) => data_from_contents(&output.name, datatype, contents)?,
            (None, None) => {
                return Err(InferError::Internal(format!(
                    "output '{}' carries no contents",
                    output.name
                )))
            }
        };

        let tensor = Tensor::new(&output.name, output.shape.clone(), data);
        tensor.validate()?;
        outputs.push(tensor);
    }

    Ok(InferResponse {
        model_name: response.model_name,
        model_version: response.model_version,
        id: (!response.id.is_empty()).then_some(response.id),
        parameters: parameters_from_proto(response.parameters),
        outputs,
    })
}

/// Reads a `ModelMetadataResponse` into domain types, for the client side.
pub fn metadata_from_proto(response: pb::ModelMetadataResponse) -> Result<ModelMetadata> {
    let tensors = |tensors: Vec<pb::model_metadata_response::TensorMetadata>| {
        tensors
            .into_iter()
            .map(|tensor| {
                Ok(galemind_core::TensorMetadata {
                    name: tensor.name,
                    datatype: DataType::from_str(&tensor.datatype)?,
                    shape: tensor.shape,
                })
            })
            .collect::<Result<Vec<_>>>()
    };

    Ok(ModelMetadata {
        name: response.name,
        versions: response.versions,
        platform: response.platform,
        inputs: tensors(response.inputs)?,
        outputs: tensors(response.outputs)?,
        // The protocol's metadata carries no batch limit: the server decides
        // it, so a client must not assume one.
        max_batch_size: None,
    })
}

/// Maps a gRPC status back onto a domain error, for the client side.
///
/// The server's own wording is kept: it knows what went wrong, and restating it
/// in the client's words would only nest one sentence inside another.
pub fn from_status(status: Status) -> InferError {
    let message = status.message().to_string();
    match status.code() {
        Code::ResourceExhausted => InferError::Overloaded,
        Code::DeadlineExceeded => InferError::Timeout(std::time::Duration::from_secs(0)),
        Code::NotFound | Code::InvalidArgument | Code::Unimplemented | Code::FailedPrecondition => {
            InferError::remote_client_error(message)
        }
        _ => InferError::remote_server_error(message),
    }
}

#[cfg(test)]
mod client_tests {
    use super::*;
    use galemind_core::{ModelRef, Parameters, RequestedOutput};

    fn request() -> InferRequest {
        InferRequest {
            model: ModelRef::new("energy", Some("2".to_string())),
            id: Some("req-1".to_string()),
            parameters: Parameters::new(),
            inputs: vec![Tensor::new(
                "features",
                vec![1, 2],
                TensorData::Fp32(vec![1.5, 2.5]),
            )],
            requested_outputs: vec![RequestedOutput {
                name: "kwh".to_string(),
                parameters: Parameters::new(),
            }],
        }
    }

    #[test]
    fn a_request_encoded_by_the_client_is_decoded_by_the_server_unchanged() {
        let original = request();
        let decoded = request_from_proto(request_to_proto(&original)).unwrap();

        assert_eq!(decoded.model, original.model);
        assert_eq!(decoded.id, original.id);
        assert_eq!(decoded.inputs, original.inputs);
        assert_eq!(decoded.requested_outputs, original.requested_outputs);
    }

    #[test]
    fn a_response_encoded_by_the_server_is_decoded_by_the_client_unchanged() {
        let original = InferResponse {
            model_name: "energy".to_string(),
            model_version: "2".to_string(),
            id: Some("req-1".to_string()),
            parameters: Parameters::new(),
            outputs: vec![
                Tensor::new("kwh", vec![2, 1], TensorData::Fp32(vec![1.0, 2.0])),
                Tensor::new("band", vec![2], TensorData::Int64(vec![0, 2])),
            ],
        };

        let decoded = response_from_proto(response_to_proto(original.clone())).unwrap();
        assert_eq!(decoded.model_name, original.model_name);
        assert_eq!(decoded.model_version, original.model_version);
        assert_eq!(decoded.id, original.id);
        assert_eq!(decoded.outputs, original.outputs);
    }

    #[test]
    fn metadata_survives_a_round_trip_through_the_protocol() {
        let original = ModelMetadata {
            name: "energy".to_string(),
            versions: vec!["1".to_string(), "2".to_string()],
            platform: "onnx_onnxv1".to_string(),
            inputs: vec![galemind_core::TensorMetadata {
                name: "features".to_string(),
                datatype: DataType::Fp32,
                shape: vec![-1, 8],
            }],
            outputs: vec![galemind_core::TensorMetadata {
                name: "kwh".to_string(),
                datatype: DataType::Fp32,
                shape: vec![-1, 1],
            }],
            max_batch_size: None,
        };

        let decoded = metadata_from_proto(metadata_to_proto(original.clone())).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn a_status_carries_the_servers_wording_and_blame_back_to_the_client() {
        for error in [
            InferError::ModelNotFound("energy".into()),
            InferError::InvalidRequest("input 'features' has rank 1".into()),
            InferError::Unsupported("nope".into()),
        ] {
            let seen = from_status(to_status(error.clone()));
            assert_eq!(seen.to_string(), error.to_string(), "wording changed");
            assert!(seen.is_client_error(), "{seen}");
        }

        for error in [
            InferError::Backend("boom".into()),
            InferError::Store("x".into()),
        ] {
            let seen = from_status(to_status(error.clone()));
            assert!(!seen.is_client_error(), "{seen}");
        }
    }

    #[test]
    fn overload_and_timeout_keep_their_meaning_across_the_wire() {
        assert!(matches!(
            from_status(to_status(InferError::Overloaded)),
            InferError::Overloaded
        ));
        assert!(from_status(to_status(InferError::Timeout(
            std::time::Duration::from_secs(5)
        )))
        .is_retryable());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    fn input(
        name: &str,
        datatype: &str,
        shape: Vec<i64>,
        contents: Option<pb::InferTensorContents>,
    ) -> pb::model_infer_request::InferInputTensor {
        pb::model_infer_request::InferInputTensor {
            name: name.to_string(),
            datatype: datatype.to_string(),
            shape,
            parameters: HashMap::new(),
            contents,
        }
    }

    fn request(
        inputs: Vec<pb::model_infer_request::InferInputTensor>,
        raw: Vec<Vec<u8>>,
    ) -> pb::ModelInferRequest {
        pb::ModelInferRequest {
            model_name: "energy".to_string(),
            model_version: String::new(),
            id: "req-1".to_string(),
            parameters: HashMap::new(),
            inputs,
            outputs: Vec::new(),
            raw_input_contents: raw,
        }
    }

    #[test]
    fn typed_contents_are_read_for_every_representable_datatype() {
        let cases: Vec<(&str, pb::InferTensorContents, TensorData)> = vec![
            (
                "BOOL",
                pb::InferTensorContents {
                    bool_contents: vec![true, false],
                    ..Default::default()
                },
                TensorData::Bool(vec![true, false]),
            ),
            (
                "INT8",
                pb::InferTensorContents {
                    int_contents: vec![-8, 8],
                    ..Default::default()
                },
                TensorData::Int8(vec![-8, 8]),
            ),
            (
                "UINT16",
                pb::InferTensorContents {
                    uint_contents: vec![1, 65_535],
                    ..Default::default()
                },
                TensorData::Uint16(vec![1, 65_535]),
            ),
            (
                "FP32",
                pb::InferTensorContents {
                    fp32_contents: vec![1.5, -2.5],
                    ..Default::default()
                },
                TensorData::Fp32(vec![1.5, -2.5]),
            ),
            (
                "BYTES",
                pb::InferTensorContents {
                    bytes_contents: vec![b"a".to_vec(), b"bc".to_vec()],
                    ..Default::default()
                },
                TensorData::Bytes(vec![b"a".to_vec(), b"bc".to_vec()]),
            ),
        ];

        for (datatype, contents, expected) in cases {
            let decoded = request_from_proto(request(
                vec![input("t", datatype, vec![2], Some(contents))],
                vec![],
            ))
            .unwrap();
            assert_eq!(decoded.inputs[0].data, expected, "{datatype}");
        }
    }

    #[test]
    fn raw_contents_are_read_and_sixteen_bit_floats_survive_exactly() {
        let data = TensorData::Fp16(vec![f16::from_f32(1.5), f16::from_f32(-0.25)]);
        let decoded = request_from_proto(request(
            vec![input("t", "FP16", vec![2], None)],
            vec![data.to_raw()],
        ))
        .unwrap();
        assert_eq!(decoded.inputs[0].data, data);
    }

    #[test]
    fn a_sixteen_bit_float_cannot_be_sent_in_typed_contents() {
        let err = request_from_proto(request(
            vec![input(
                "t",
                "BF16",
                vec![1],
                Some(pb::InferTensorContents {
                    fp32_contents: vec![1.0],
                    ..Default::default()
                }),
            )],
            vec![],
        ))
        .unwrap_err();
        assert!(err.to_string().contains("raw_input_contents"), "{err}");
    }

    #[test]
    fn a_value_outside_the_narrow_types_range_is_rejected() {
        let err = request_from_proto(request(
            vec![input(
                "t",
                "INT8",
                vec![1],
                Some(pb::InferTensorContents {
                    int_contents: vec![1_000],
                    ..Default::default()
                }),
            )],
            vec![],
        ))
        .unwrap_err();
        assert!(
            err.to_string().contains("outside the range of INT8"),
            "{err}"
        );
    }

    #[test]
    fn sending_both_representations_is_rejected() {
        let err = request_from_proto(request(
            vec![input(
                "t",
                "FP32",
                vec![1],
                Some(pb::InferTensorContents {
                    fp32_contents: vec![1.0],
                    ..Default::default()
                }),
            )],
            vec![TensorData::Fp32(vec![1.0]).to_raw()],
        ))
        .unwrap_err();
        assert!(err.to_string().contains("both contents"), "{err}");
    }

    #[test]
    fn a_mismatched_number_of_raw_entries_is_rejected() {
        let err = request_from_proto(request(
            vec![input("t", "FP32", vec![1], None)],
            vec![vec![0u8; 4], vec![0u8; 4]],
        ))
        .unwrap_err();
        assert!(err.to_string().contains("for 1 inputs"), "{err}");
    }

    #[test]
    fn contents_that_disagree_with_the_shape_are_rejected() {
        let err = request_from_proto(request(
            vec![input(
                "t",
                "FP32",
                vec![2, 2],
                Some(pb::InferTensorContents {
                    fp32_contents: vec![1.0],
                    ..Default::default()
                }),
            )],
            vec![],
        ))
        .unwrap_err();
        assert!(err.to_string().contains("4 elements"), "{err}");
    }

    #[test]
    fn a_variable_dimension_is_not_accepted_in_a_request() {
        let err = request_from_proto(request(vec![input("t", "FP32", vec![-1, 4], None)], vec![]))
            .unwrap_err();
        assert!(err.to_string().contains("concrete"), "{err}");
    }

    #[test]
    fn an_empty_request_id_means_no_id() {
        let mut message = request(
            vec![input(
                "t",
                "FP32",
                vec![1],
                Some(pb::InferTensorContents {
                    fp32_contents: vec![1.0],
                    ..Default::default()
                }),
            )],
            vec![],
        );
        message.id = String::new();
        assert_eq!(request_from_proto(message).unwrap().id, None);
    }

    #[test]
    fn parameters_round_trip_through_the_protocols_three_value_kinds() {
        let mut message = request(
            vec![input(
                "t",
                "FP32",
                vec![1],
                Some(pb::InferTensorContents {
                    fp32_contents: vec![1.0],
                    ..Default::default()
                }),
            )],
            vec![],
        );
        message.parameters = HashMap::from([
            (
                "flag".to_string(),
                pb::InferParameter {
                    parameter_choice: Some(pb::infer_parameter::ParameterChoice::BoolParam(true)),
                },
            ),
            (
                "count".to_string(),
                pb::InferParameter {
                    parameter_choice: Some(pb::infer_parameter::ParameterChoice::Int64Param(7)),
                },
            ),
            (
                "tag".to_string(),
                pb::InferParameter {
                    parameter_choice: Some(pb::infer_parameter::ParameterChoice::StringParam(
                        "x".to_string(),
                    )),
                },
            ),
        ]);

        let decoded = request_from_proto(message).unwrap();
        assert_eq!(decoded.parameters["flag"], Parameter::Bool(true));
        assert_eq!(decoded.parameters["count"], Parameter::Int64(7));
        assert_eq!(decoded.parameters["tag"], Parameter::String("x".into()));

        let encoded = parameters_to_proto(&decoded.parameters);
        assert_eq!(encoded.len(), 3);
    }

    #[test]
    fn a_response_is_encoded_with_raw_contents_and_no_typed_contents() {
        let response = InferResponse {
            model_name: "energy".to_string(),
            model_version: "1".to_string(),
            id: Some("req-1".to_string()),
            parameters: Parameters::new(),
            outputs: vec![Tensor::new(
                "kwh",
                vec![2, 1],
                TensorData::Fp32(vec![1.0, 2.0]),
            )],
        };

        let encoded = response_to_proto(response);
        assert_eq!(encoded.id, "req-1");
        assert_eq!(encoded.outputs[0].datatype, "FP32");
        assert_eq!(encoded.outputs[0].shape, vec![2, 1]);
        assert!(encoded.outputs[0].contents.is_none());
        assert_eq!(encoded.raw_output_contents.len(), 1);
        assert_eq!(
            TensorData::from_raw(DataType::Fp32, &encoded.raw_output_contents[0], 2).unwrap(),
            TensorData::Fp32(vec![1.0, 2.0])
        );
    }

    #[test]
    fn errors_map_onto_the_expected_grpc_codes() {
        let cases = [
            (InferError::ModelNotFound("m".into()), Code::NotFound),
            (
                InferError::InvalidRequest("bad".into()),
                Code::InvalidArgument,
            ),
            (InferError::Overloaded, Code::ResourceExhausted),
            (
                InferError::Timeout(std::time::Duration::from_secs(1)),
                Code::DeadlineExceeded,
            ),
            (InferError::Backend("boom".into()), Code::Internal),
            (InferError::Unsupported("nope".into()), Code::Unimplemented),
        ];
        for (error, expected) in cases {
            assert_eq!(to_status(error).code(), expected);
        }
    }
}
