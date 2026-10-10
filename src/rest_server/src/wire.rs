//! The JSON objects defined by the Predict Protocol v2.
//!
//! Field names are exactly those in the specification, which is why none of
//! these types rename anything: `model_name` is `model_name` on the wire.

use galemind_core::{InferResponse, ModelMetadata, Parameter, Parameters, Tensor};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Zero or more parameters, each a string, number or boolean.
pub type JsonParameters = BTreeMap<String, Value>;

/// `$metadata_server_response`
#[derive(Debug, Serialize, Deserialize)]
pub struct ServerMetadata {
    pub name: String,
    pub version: String,
    pub extensions: Vec<String>,
}

/// `$metadata_tensor`
#[derive(Debug, Serialize, Deserialize)]
pub struct MetadataTensor {
    pub name: String,
    pub datatype: String,
    pub shape: Vec<i64>,
}

/// `$metadata_model_response`
#[derive(Debug, Serialize, Deserialize)]
pub struct ModelMetadataBody {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub versions: Option<Vec<String>>,
    pub platform: String,
    pub inputs: Vec<MetadataTensor>,
    pub outputs: Vec<MetadataTensor>,
}

impl From<ModelMetadata> for ModelMetadataBody {
    fn from(metadata: ModelMetadata) -> Self {
        fn tensors(tensors: Vec<galemind_core::TensorMetadata>) -> Vec<MetadataTensor> {
            tensors
                .into_iter()
                .map(|tensor| MetadataTensor {
                    name: tensor.name,
                    datatype: tensor.datatype.to_string(),
                    shape: tensor.shape,
                })
                .collect()
        }

        Self {
            name: metadata.name,
            versions: (!metadata.versions.is_empty()).then_some(metadata.versions),
            platform: metadata.platform,
            inputs: tensors(metadata.inputs),
            outputs: tensors(metadata.outputs),
        }
    }
}

/// `$request_input`
#[derive(Debug, Serialize, Deserialize)]
pub struct RequestInput {
    pub name: String,
    pub shape: Vec<i64>,
    pub datatype: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<JsonParameters>,
    pub data: Value,
}

/// `$request_output`
#[derive(Debug, Serialize, Deserialize)]
pub struct RequestOutput {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<JsonParameters>,
}

/// `$inference_request`
#[derive(Debug, Serialize, Deserialize)]
pub struct InferenceRequestBody {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<JsonParameters>,
    pub inputs: Vec<RequestInput>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outputs: Option<Vec<RequestOutput>>,
}

/// `$response_output`
#[derive(Debug, Serialize, Deserialize)]
pub struct ResponseOutput {
    pub name: String,
    pub shape: Vec<i64>,
    pub datatype: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<JsonParameters>,
    pub data: Value,
}

/// `$inference_response`
#[derive(Debug, Serialize, Deserialize)]
pub struct InferenceResponseBody {
    pub model_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<JsonParameters>,
    pub outputs: Vec<ResponseOutput>,
}

/// `$inference_error_response` and the other error objects, which share a shape.
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorBody {
    pub error: String,
}

/// Encodes domain parameters as JSON values.
pub fn parameters_to_json(parameters: &Parameters) -> Option<JsonParameters> {
    if parameters.is_empty() {
        return None;
    }
    Some(
        parameters
            .iter()
            .map(|(name, value)| {
                let value = match value {
                    Parameter::Bool(v) => Value::Bool(*v),
                    Parameter::Int64(v) => Value::from(*v),
                    Parameter::String(v) => Value::String(v.clone()),
                };
                (name.clone(), value)
            })
            .collect(),
    )
}

/// Builds the response body, encoding each output's contents for its datatype.
pub fn response_to_json(
    response: InferResponse,
) -> Result<InferenceResponseBody, galemind_core::InferError> {
    let outputs = response
        .outputs
        .into_iter()
        .map(|tensor| {
            let datatype = tensor.data.datatype().to_string();
            let Tensor { name, shape, data } = tensor;
            let (data, encoding) = crate::tensor_json::encode(&data);

            // Only say how bytes were encoded when it is not plain text, so
            // ordinary responses carry no server-specific parameters at all.
            let parameters = match encoding {
                crate::tensor_json::BytesEncoding::Base64 => Some(
                    [(
                        crate::tensor_json::BINARY_ENCODING_PARAMETER.to_string(),
                        Value::String("base64".to_string()),
                    )]
                    .into_iter()
                    .collect(),
                ),
                crate::tensor_json::BytesEncoding::Text => None,
            };

            Ok(ResponseOutput {
                name,
                shape,
                datatype,
                parameters,
                data,
            })
        })
        .collect::<Result<Vec<_>, galemind_core::InferError>>()?;

    Ok(InferenceResponseBody {
        model_name: response.model_name,
        model_version: Some(response.model_version),
        id: response.id,
        parameters: parameters_to_json(&response.parameters),
        outputs,
    })
}
