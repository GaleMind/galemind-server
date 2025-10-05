use crate::data_model::{InferenceRequest as RestInferenceRequest, InferenceResponse as RestInferenceResponse, MetadataTensor, TensorData};
use foundation::{InferenceRequest as FoundationInferenceRequest, InferenceResponse as FoundationInferenceResponse};
use foundation::api::inference::{InferenceOutput, InferenceError, InferParameter};
use foundation::api::tensor::{Data, DataType};
use std::collections::HashMap;

pub fn convert_rest_to_foundation(rest_request: RestInferenceRequest) -> FoundationInferenceRequest {
    let mut inference_outputs = Vec::new();

    for input_tensor in rest_request.inputs {
        let data = match input_tensor.data {
            Some(TensorData::Int32(data)) => Data::Int32(data),
            Some(TensorData::Int64(data)) => Data::Int64(data),
            Some(TensorData::Float32(data)) => Data::Float32(data),
            Some(TensorData::Float64(data)) => Data::Float64(data),
            Some(TensorData::Bool(data)) => Data::Bool(data),
            None => Data::Float32(vec![]), // Default empty
        };

        let datatype = match input_tensor.datatype.as_str() {
            "FP32" | "FLOAT32" => DataType::Float32,
            "FP64" | "FLOAT64" => DataType::Float64,
            "INT32" => DataType::Int32,
            "INT64" => DataType::Int64,
            "BOOL" => DataType::Bool,
            _ => DataType::Float32, // Default
        };

        let shape: Vec<usize> = input_tensor.shape.iter().map(|&x| x as usize).collect();

        let parameters = input_tensor.parameters.map(|params| {
            params.into_iter().map(|(k, v)| {
                let param = match v {
                    serde_json::Value::Bool(b) => InferParameter::Bool(b),
                    serde_json::Value::Number(n) => {
                        if let Some(i) = n.as_i64() {
                            InferParameter::Int64(i)
                        } else if let Some(f) = n.as_f64() {
                            InferParameter::Double(f)
                        } else {
                            InferParameter::Double(0.0)
                        }
                    },
                    serde_json::Value::String(s) => InferParameter::String(s),
                    _ => InferParameter::String(v.to_string()),
                };
                (k, param)
            }).collect()
        });

        inference_outputs.push(InferenceOutput {
            name: input_tensor.name,
            shape,
            datatype,
            parameters,
            data,
        });
    }

    FoundationInferenceRequest {
        model_name: "default_model".to_string(), // TODO: Extract from path
        model_version: None,
        id: rest_request.id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
        parameters: rest_request.parameters.map(|params| {
            params.into_iter().map(|(k, v)| {
                let param = match v {
                    serde_json::Value::Bool(b) => InferParameter::Bool(b),
                    serde_json::Value::Number(n) => {
                        if let Some(i) = n.as_i64() {
                            InferParameter::Int64(i)
                        } else if let Some(f) = n.as_f64() {
                            InferParameter::Double(f)
                        } else {
                            InferParameter::Double(0.0)
                        }
                    },
                    serde_json::Value::String(s) => InferParameter::String(s),
                    _ => InferParameter::String(v.to_string()),
                };
                (k, param)
            }).collect()
        }),
        outputs: Some(inference_outputs),
    }
}

pub fn convert_foundation_to_rest(foundation_response: FoundationInferenceResponse) -> RestInferenceResponse {
    match foundation_response {
        FoundationInferenceResponse::Ok(output) => {
            let tensor_data = match output.data {
                Data::Float32(data) => TensorData::Float32(data),
                Data::Float64(data) => TensorData::Float64(data),
                Data::Int32(data) => TensorData::Int32(data),
                Data::Int64(data) => TensorData::Int64(data),
                Data::Bool(data) => TensorData::Bool(data),
                Data::VFLOAT(data) => TensorData::Float64(data),
            };

            let datatype = match output.datatype {
                DataType::Float32 => "FP32".to_string(),
                DataType::Float64 => "FP64".to_string(),
                DataType::Int32 => "INT32".to_string(),
                DataType::Int64 => "INT64".to_string(),
                DataType::Bool => "BOOL".to_string(),
                DataType::VFLOAT => "FP64".to_string(),
            };

            let shape: Vec<u64> = output.shape.iter().map(|&x| x as u64).collect();

            let parameters = output.parameters.map(|params| {
                params.into_iter().map(|(k, v)| {
                    let json_value = match v {
                        InferParameter::Bool(b) => serde_json::Value::Bool(b),
                        InferParameter::Int64(i) => serde_json::Value::Number(serde_json::Number::from(i)),
                        InferParameter::Double(f) => serde_json::Value::Number(
                            serde_json::Number::from_f64(f).unwrap_or_else(|| serde_json::Number::from(0))
                        ),
                        InferParameter::String(s) => serde_json::Value::String(s),
                    };
                    (k, json_value)
                }).collect()
            });

            RestInferenceResponse {
                id: None,
                outputs: Some(vec![MetadataTensor {
                    name: output.name,
                    shape,
                    datatype,
                    parameters,
                    data: Some(tensor_data),
                }]),
            }
        },
        FoundationInferenceResponse::Error(error) => {
            // Return a response with empty outputs in case of error
            // In a production system, you might want to handle this differently
            RestInferenceResponse {
                id: None,
                outputs: Some(vec![]),
            }
        }
    }
}