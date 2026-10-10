//! The JSON encoding defined by the Predict Protocol v2.
//!
//! Field names are exactly those in the specification, which is why none of
//! these types rename anything: `model_name` is `model_name` on the wire.
//!
//! This lives beside the domain rather than inside the HTTP server because the
//! encoding is part of the protocol, not of one side of it: the server and the
//! client both speak it, and sharing it means a change can only be made in one
//! place.

pub mod tensor;

pub use tensor::{decode, encode, BytesEncoding, BINARY_ENCODING_PARAMETER};

use crate::catalogue::{CatalogueEntry, ModelInfo, VersionInfo};
use crate::error::InferError;
use crate::infer::{InferResponse, Parameter, Parameters};
use crate::model::{ModelMetadata, TensorMetadata};
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Zero or more parameters, each a string, number or boolean.
pub type JsonParameters = BTreeMap<String, Value>;

/// `$metadata_server_response`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServerMetadataBody {
    pub name: String,
    pub version: String,
    pub extensions: Vec<String>,
}

/// `GET /v2/models/list`: every model the server can serve.
///
/// Not part of the Predict Protocol v2, which has no catalogue call. The
/// shape is this server's own, and `count` is included so a caller can check
/// a truncated read without walking the array.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelListBody {
    pub count: usize,
    pub models: Vec<ModelListEntry>,
}

/// One model in `GET /v2/models/list`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelListEntry {
    pub name: String,
    /// Oldest first, in the order the server resolves them.
    pub versions: Vec<String>,
    /// The version an unversioned request resolves to.
    pub latest_version: String,
    pub platform: String,
    /// Versions in memory now. Models load lazily, so this changes with
    /// traffic rather than describing the repository.
    pub loaded_versions: Vec<String>,
    /// Whether a request that names no version would be served without a load.
    pub ready: bool,
}

impl From<CatalogueEntry> for ModelListEntry {
    fn from(entry: CatalogueEntry) -> Self {
        Self {
            ready: entry.latest_is_loaded(),
            name: entry.name,
            versions: entry.versions,
            latest_version: entry.latest_version,
            platform: entry.platform,
            loaded_versions: entry.loaded_versions,
        }
    }
}

/// `GET /v2/models/info/{model_name}`: every version of one model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelInfoBody {
    pub name: String,
    /// The version an unversioned request resolves to.
    pub latest_version: String,
    pub versions: Vec<ModelVersionBody>,
}

/// One version in `GET /v2/models/info/{model_name}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelVersionBody {
    pub version: String,
    pub platform: String,
    pub digest: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_batch_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub intra_op_threads: Option<usize>,
    pub loaded: bool,
    pub is_latest: bool,
    /// The signature as the runtime reports it. Empty for a version that is
    /// not in memory and was not loaded for this call.
    pub inputs: Vec<MetadataTensor>,
    pub outputs: Vec<MetadataTensor>,
}

impl From<VersionInfo> for ModelVersionBody {
    fn from(version: VersionInfo) -> Self {
        Self {
            version: version.version,
            platform: version.platform,
            digest: version.digest,
            size_bytes: version.size_bytes,
            max_batch_size: version.max_batch_size,
            intra_op_threads: version.intra_op_threads,
            loaded: version.loaded,
            is_latest: version.is_latest,
            inputs: metadata_tensors(version.inputs),
            outputs: metadata_tensors(version.outputs),
        }
    }
}

impl From<ModelInfo> for ModelInfoBody {
    fn from(info: ModelInfo) -> Self {
        Self {
            name: info.name,
            latest_version: info.latest_version,
            versions: info.versions.into_iter().map(Into::into).collect(),
        }
    }
}

/// Tensor metadata in the protocol's JSON shape.
fn metadata_tensors(tensors: Vec<TensorMetadata>) -> Vec<MetadataTensor> {
    tensors
        .into_iter()
        .map(|tensor| MetadataTensor {
            name: tensor.name,
            datatype: tensor.datatype.to_string(),
            shape: tensor.shape,
        })
        .collect()
}

/// `$metadata_tensor`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetadataTensor {
    pub name: String,
    pub datatype: String,
    pub shape: Vec<i64>,
}

/// `$metadata_model_response`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
        Self {
            name: metadata.name,
            versions: (!metadata.versions.is_empty()).then_some(metadata.versions),
            platform: metadata.platform,
            inputs: metadata_tensors(metadata.inputs),
            outputs: metadata_tensors(metadata.outputs),
        }
    }
}

/// `$request_input`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestInput {
    pub name: String,
    pub shape: Vec<i64>,
    pub datatype: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<JsonParameters>,
    pub data: Value,
}

/// `$request_output`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestOutput {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<JsonParameters>,
}

/// `$inference_request`
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseOutput {
    pub name: String,
    pub shape: Vec<i64>,
    pub datatype: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<JsonParameters>,
    pub data: Value,
}

/// `$inference_response`
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
pub fn response_to_json(response: InferResponse) -> Result<InferenceResponseBody, InferError> {
    let outputs = response
        .outputs
        .into_iter()
        .map(|tensor| {
            let datatype = tensor.data.datatype().to_string();
            let Tensor { name, shape, data } = tensor;
            let (data, encoding) = tensor::encode(&data);

            // Only say how bytes were encoded when it is not plain text, so
            // ordinary responses carry no server-specific parameters at all.
            let parameters = match encoding {
                tensor::BytesEncoding::Base64 => Some(
                    [(
                        tensor::BINARY_ENCODING_PARAMETER.to_string(),
                        Value::String("base64".to_string()),
                    )]
                    .into_iter()
                    .collect(),
                ),
                tensor::BytesEncoding::Text => None,
            };

            Ok(ResponseOutput {
                name,
                shape,
                datatype,
                parameters,
                data,
            })
        })
        .collect::<Result<Vec<_>, InferError>>()?;

    Ok(InferenceResponseBody {
        model_name: response.model_name,
        model_version: Some(response.model_version),
        id: response.id,
        parameters: parameters_to_json(&response.parameters),
        outputs,
    })
}
