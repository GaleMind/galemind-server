//! The HTTP/REST client.

use crate::PredictV2Client;
use async_trait::async_trait;
use galemind_core::json::{
    self, ErrorBody, InferenceRequestBody, InferenceResponseBody, ModelMetadataBody, RequestInput,
    RequestOutput, ServerMetadataBody,
};
use galemind_core::{
    DataType, InferError, InferRequest, InferResponse, ModelMetadata, ModelRef, Parameter,
    ServerMetadata, Tensor, TensorMetadata,
};
use reqwest::{Client, Response, StatusCode};
use serde::de::DeserializeOwned;
use std::str::FromStr;
use std::time::Duration;

type Result<T> = std::result::Result<T, InferError>;

/// Talks the protocol over HTTP.
pub struct HttpClient {
    client: Client,
    base: String,
}

impl HttpClient {
    /// `endpoint` is the server's base URL, for example `http://localhost:8000`.
    pub fn new(endpoint: &str) -> Result<Self> {
        Self::with_timeout(endpoint, Duration::from_secs(30))
    }

    pub fn with_timeout(endpoint: &str, timeout: Duration) -> Result<Self> {
        let client = Client::builder()
            .timeout(timeout)
            // Inference clients tend to be chatty; reusing connections matters
            // far more than it does for a browser.
            .pool_max_idle_per_host(64)
            .build()
            .map_err(|e| InferError::Internal(format!("cannot build an HTTP client: {e}")))?;

        Ok(Self {
            client,
            base: endpoint.trim_end_matches('/').to_string(),
        })
    }

    /// The base URL this client talks to.
    pub fn endpoint(&self) -> &str {
        &self.base
    }

    /// The path of a model, with its version when one is given.
    fn model_path(&self, model: &ModelRef, suffix: &str) -> String {
        match &model.version {
            Some(version) => format!(
                "{}/v2/models/{}/versions/{version}{suffix}",
                self.base, model.name
            ),
            None => format!("{}/v2/models/{}{suffix}", self.base, model.name),
        }
    }

    /// Fetches a URL whose answer is a JSON body.
    async fn get_json<T: DeserializeOwned>(&self, url: &str) -> Result<T> {
        let response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| transport_error(url, e))?;
        read_json(response).await
    }

    /// Reads a health endpoint, where the status code is the whole answer.
    async fn health(&self, url: &str) -> Result<bool> {
        let response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| transport_error(url, e))?;

        match response.status() {
            StatusCode::OK => Ok(true),
            status if status.is_client_error() => Ok(false),
            status => Err(InferError::Internal(format!(
                "{url} answered {status}, which is neither ready nor not-ready"
            ))),
        }
    }
}

fn transport_error(url: &str, error: reqwest::Error) -> InferError {
    if error.is_timeout() {
        return InferError::Timeout(Duration::from_secs(0));
    }
    InferError::Internal(format!("{url}: {error}"))
}

/// Reads a JSON body, turning the protocol's error object into a domain error.
async fn read_json<T: DeserializeOwned>(response: Response) -> Result<T> {
    let status = response.status();
    let bytes = response
        .bytes()
        .await
        .map_err(|e| InferError::Internal(format!("cannot read the response body: {e}")))?;

    if !status.is_success() {
        let message = serde_json::from_slice::<ErrorBody>(&bytes)
            .map(|body| body.error)
            .unwrap_or_else(|_| String::from_utf8_lossy(&bytes).into_owned());

        return Err(match status {
            StatusCode::SERVICE_UNAVAILABLE => InferError::Overloaded,
            StatusCode::GATEWAY_TIMEOUT => InferError::Timeout(Duration::from_secs(0)),
            status if status.is_client_error() => InferError::remote_client_error(message),
            _ => InferError::remote_server_error(message),
        });
    }

    serde_json::from_slice(&bytes).map_err(|e| {
        InferError::Internal(format!(
            "cannot read the response body as the protocol's JSON: {e}"
        ))
    })
}

/// Builds the request body from a domain request.
fn to_body(request: &InferRequest) -> Result<InferenceRequestBody> {
    let mut inputs = Vec::with_capacity(request.inputs.len());
    for tensor in &request.inputs {
        let (data, encoding) = json::encode(&tensor.data);
        let parameters = match encoding {
            json::BytesEncoding::Base64 => Some(
                [(
                    json::BINARY_ENCODING_PARAMETER.to_string(),
                    serde_json::Value::String("base64".to_string()),
                )]
                .into_iter()
                .collect(),
            ),
            json::BytesEncoding::Text => None,
        };

        inputs.push(RequestInput {
            name: tensor.name.clone(),
            shape: tensor.shape.clone(),
            datatype: tensor.data.datatype().to_string(),
            parameters,
            data,
        });
    }

    Ok(InferenceRequestBody {
        id: request.id.clone(),
        parameters: json::parameters_to_json(&request.parameters),
        inputs,
        outputs: (!request.requested_outputs.is_empty()).then(|| {
            request
                .requested_outputs
                .iter()
                .map(|output| RequestOutput {
                    name: output.name.clone(),
                    parameters: json::parameters_to_json(&output.parameters),
                })
                .collect()
        }),
    })
}

/// Reads a response body back into domain types.
fn from_body(body: InferenceResponseBody) -> Result<InferResponse> {
    let mut outputs = Vec::with_capacity(body.outputs.len());
    for output in body.outputs {
        let datatype = DataType::from_str(&output.datatype)?;
        let encoding = json::BytesEncoding::from_parameters(output.parameters.as_ref())?;
        let tensor = Tensor::new(
            &output.name,
            output.shape.clone(),
            json::decode(&output.name, datatype, &output.data, encoding)?,
        );
        tensor.validate()?;
        outputs.push(tensor);
    }

    Ok(InferResponse {
        model_name: body.model_name,
        model_version: body.model_version.unwrap_or_default(),
        id: body.id,
        parameters: body
            .parameters
            .into_iter()
            .flatten()
            .filter_map(|(name, value)| {
                let value = match value {
                    serde_json::Value::Bool(v) => Parameter::Bool(v),
                    serde_json::Value::String(v) => Parameter::String(v),
                    serde_json::Value::Number(v) => Parameter::Int64(v.as_i64()?),
                    _ => return None,
                };
                Some((name, value))
            })
            .collect(),
        outputs,
    })
}

fn metadata_from_body(body: ModelMetadataBody) -> Result<ModelMetadata> {
    fn tensors(tensors: Vec<json::MetadataTensor>) -> Result<Vec<TensorMetadata>> {
        tensors
            .into_iter()
            .map(|tensor| {
                Ok(TensorMetadata {
                    name: tensor.name,
                    datatype: DataType::from_str(&tensor.datatype)?,
                    shape: tensor.shape,
                })
            })
            .collect()
    }

    Ok(ModelMetadata {
        name: body.name,
        versions: body.versions.unwrap_or_default(),
        platform: body.platform,
        inputs: tensors(body.inputs)?,
        outputs: tensors(body.outputs)?,
        // The protocol's metadata does not report a batch limit; the server
        // decides it, so a client must not assume one.
        max_batch_size: None,
    })
}

#[async_trait]
impl PredictV2Client for HttpClient {
    async fn server_live(&self) -> Result<bool> {
        self.health(&format!("{}/v2/health/live", self.base)).await
    }

    async fn server_ready(&self) -> Result<bool> {
        self.health(&format!("{}/v2/health/ready", self.base)).await
    }

    async fn model_ready(&self, model: &ModelRef) -> Result<bool> {
        self.health(&self.model_path(model, "/ready")).await
    }

    async fn server_metadata(&self) -> Result<ServerMetadata> {
        let body: ServerMetadataBody = self.get_json(&format!("{}/v2", self.base)).await?;
        Ok(ServerMetadata {
            name: body.name,
            version: body.version,
            extensions: body.extensions,
        })
    }

    async fn model_metadata(&self, model: &ModelRef) -> Result<ModelMetadata> {
        let body: ModelMetadataBody = self.get_json(&self.model_path(model, "")).await?;
        metadata_from_body(body)
    }

    async fn infer(&self, request: InferRequest) -> Result<InferResponse> {
        let url = self.model_path(&request.model, "/infer");
        let response = self
            .client
            .post(&url)
            .json(&to_body(&request)?)
            .send()
            .await
            .map_err(|e| transport_error(&url, e))?;

        from_body(read_json(response).await?)
    }

    fn transport(&self) -> &'static str {
        "http"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use galemind_core::{Parameters, RequestedOutput, TensorData};

    fn request() -> InferRequest {
        InferRequest {
            model: ModelRef::latest("energy"),
            id: Some("req-1".to_string()),
            parameters: Parameters::new(),
            inputs: vec![Tensor::new(
                "features",
                vec![1, 2],
                TensorData::Fp32(vec![1.0, 2.0]),
            )],
            requested_outputs: vec![RequestedOutput {
                name: "kwh".to_string(),
                parameters: Parameters::new(),
            }],
        }
    }

    #[test]
    fn urls_follow_the_protocol_with_and_without_a_version() {
        let client = HttpClient::new("http://localhost:8000/").unwrap();
        assert_eq!(client.endpoint(), "http://localhost:8000");
        assert_eq!(
            client.model_path(&ModelRef::latest("energy"), "/infer"),
            "http://localhost:8000/v2/models/energy/infer"
        );
        assert_eq!(
            client.model_path(&ModelRef::new("energy", Some("3".into())), "/ready"),
            "http://localhost:8000/v2/models/energy/versions/3/ready"
        );
    }

    #[test]
    fn a_request_is_encoded_as_the_protocols_body() {
        let body = to_body(&request()).unwrap();
        assert_eq!(body.id.as_deref(), Some("req-1"));
        assert_eq!(body.inputs[0].name, "features");
        assert_eq!(body.inputs[0].datatype, "FP32");
        assert_eq!(body.inputs[0].shape, vec![1, 2]);
        assert_eq!(body.inputs[0].data, serde_json::json!([1.0, 2.0]));
        assert_eq!(body.outputs.unwrap()[0].name, "kwh");
    }

    #[test]
    fn a_request_without_requested_outputs_omits_the_field() {
        let mut request = request();
        request.requested_outputs.clear();
        assert!(to_body(&request).unwrap().outputs.is_none());
    }

    #[test]
    fn a_response_body_is_read_back_into_domain_tensors() {
        let body: InferenceResponseBody = serde_json::from_value(serde_json::json!({
            "model_name": "energy",
            "model_version": "1",
            "id": "req-1",
            "outputs": [
                {"name": "kwh", "shape": [2, 1], "datatype": "FP32", "data": [1.5, 2.5]}
            ]
        }))
        .unwrap();

        let response = from_body(body).unwrap();
        assert_eq!(response.model_name, "energy");
        assert_eq!(response.model_version, "1");
        assert_eq!(response.id.as_deref(), Some("req-1"));
        assert_eq!(response.outputs[0].data, TensorData::Fp32(vec![1.5, 2.5]));
    }

    #[test]
    fn a_response_whose_data_contradicts_its_shape_is_rejected() {
        let body: InferenceResponseBody = serde_json::from_value(serde_json::json!({
            "model_name": "energy",
            "outputs": [
                {"name": "kwh", "shape": [4, 1], "datatype": "FP32", "data": [1.5]}
            ]
        }))
        .unwrap();
        assert!(from_body(body).is_err());
    }

    #[test]
    fn base64_bytes_survive_the_round_trip_through_the_client() {
        let mut request = request();
        request.inputs = vec![Tensor::new(
            "raw",
            vec![1],
            TensorData::Bytes(vec![vec![0xff, 0x00]]),
        )];

        let body = to_body(&request).unwrap();
        let parameters = body.inputs[0].parameters.as_ref().unwrap();
        assert_eq!(
            parameters[json::BINARY_ENCODING_PARAMETER],
            serde_json::Value::String("base64".to_string())
        );
    }

    #[test]
    fn model_metadata_is_read_back_with_its_datatypes_parsed() {
        let body: ModelMetadataBody = serde_json::from_value(serde_json::json!({
            "name": "energy",
            "versions": ["1"],
            "platform": "onnx_onnxv1",
            "inputs": [{"name": "features", "datatype": "FP32", "shape": [-1, 8]}],
            "outputs": [{"name": "kwh", "datatype": "FP32", "shape": [-1, 1]}]
        }))
        .unwrap();

        let metadata = metadata_from_body(body).unwrap();
        assert_eq!(metadata.inputs[0].datatype, DataType::Fp32);
        assert_eq!(metadata.inputs[0].shape, vec![-1, 8]);
        assert!(metadata.inputs[0].has_dynamic_batch_dim());
    }
}
