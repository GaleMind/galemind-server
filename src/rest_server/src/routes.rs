//! The eight URLs the Predict Protocol v2 defines over HTTP.

use crate::error::ApiError;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use galemind_core::json::{
    self, tensor as tensor_json, ErrorBody, InferenceRequestBody, InferenceResponseBody,
    JsonParameters, ModelInfoBody, ModelListBody, ModelListEntry, ModelMetadataBody,
    ServerMetadataBody,
};
use galemind_core::{
    DataType, InferError, InferRequest, Inference, MetricsSource, ModelRef, Parameter, Parameters,
    RequestedOutput, Tensor, SERVER_NAME, SERVER_VERSION,
};
use serde::Deserialize;
use serde_json::Value;
use std::str::FromStr;
use std::sync::Arc;
use tracing::debug;

/// What every handler shares.
pub type AppState = Arc<dyn Inference>;

/// Builds the router for the protocol's URLs, and for the two catalogue URLs
/// this server adds to them.
///
/// The version is a literal path segment, not a parameter: a request to
/// `/v3/health/live` is a 404, where capturing the segment would have answered
/// any version at all.
///
/// `list` and `info` are therefore reserved model names: a static segment
/// wins over a parameter, so `/v2/models/list` is the catalogue and never the
/// metadata of a model called `list`. Nothing else is shadowed, and the
/// protocol's own URLs are untouched.
pub fn router(inference: AppState) -> Router {
    Router::new()
        .route("/v2", get(server_metadata))
        .route("/v2/health/live", get(server_live))
        .route("/v2/health/ready", get(server_ready))
        .route("/v2/models/list", get(model_list))
        .route("/v2/models/info/{model_name}", get(model_info))
        .route("/v2/models/{model_name}", get(model_metadata))
        .route("/v2/models/{model_name}/ready", get(model_ready))
        .route("/v2/models/{model_name}/infer", post(model_infer))
        .route(
            "/v2/models/{model_name}/versions/{model_version}",
            get(model_version_metadata),
        )
        .route(
            "/v2/models/{model_name}/versions/{model_version}/ready",
            get(model_version_ready),
        )
        .route(
            "/v2/models/{model_name}/versions/{model_version}/infer",
            post(model_version_infer),
        )
        .with_state(inference)
}

/// Health responses carry no body, as the specification requires: the status
/// code alone is the answer, 200 for true and 4xx for false.
fn health(ready: bool) -> StatusCode {
    if ready {
        StatusCode::OK
    } else {
        StatusCode::BAD_REQUEST
    }
}

async fn server_live() -> StatusCode {
    // Answering at all is the proof.
    StatusCode::OK
}

async fn server_ready(State(inference): State<AppState>) -> StatusCode {
    health(inference.is_ready().await)
}

async fn model_ready(
    State(inference): State<AppState>,
    Path(model_name): Path<String>,
) -> StatusCode {
    health(
        inference
            .is_model_ready(&ModelRef::latest(model_name))
            .await,
    )
}

async fn model_version_ready(
    State(inference): State<AppState>,
    Path((model_name, model_version)): Path<(String, String)>,
) -> StatusCode {
    health(
        inference
            .is_model_ready(&ModelRef::new(model_name, Some(model_version)))
            .await,
    )
}

async fn server_metadata() -> Json<ServerMetadataBody> {
    Json(ServerMetadataBody {
        name: SERVER_NAME.to_string(),
        version: SERVER_VERSION.to_string(),
        // No protocol extensions are implemented.
        extensions: Vec::new(),
    })
}

/// `GET /v2/models/list`
///
/// Loads nothing: a repository with more models than the cache holds can be
/// listed without evicting a model that requests are using.
async fn model_list(State(inference): State<AppState>) -> Result<Json<ModelListBody>, ApiError> {
    let entries: Vec<ModelListEntry> = inference
        .catalogue()
        .await?
        .into_iter()
        .map(Into::into)
        .collect();

    debug!(models = entries.len(), "listed models");
    Ok(Json(ModelListBody {
        count: entries.len(),
        models: entries,
    }))
}

/// What `GET /v2/models/info/{model_name}` accepts.
#[derive(Debug, Default, Deserialize)]
pub struct ModelInfoQuery {
    /// Load every version that is not already loaded, so that each reports its
    /// input and output signature.
    ///
    /// On by default, because a signature is what the call is for. A caller
    /// that only wants the versions and their settings passes `?load=false`
    /// and pays no loads -- worth doing against a server whose models are
    /// large and whose cache is smaller than the model being described.
    #[serde(default = "load_by_default")]
    load: bool,
}

fn load_by_default() -> bool {
    true
}

/// `GET /v2/models/info/{model_name}`
///
/// Every version of one model: its settings, its digest, whether it is
/// loaded, which version an unversioned request resolves to, and each
/// version's signature.
async fn model_info(
    State(inference): State<AppState>,
    Path(model_name): Path<String>,
    Query(query): Query<ModelInfoQuery>,
) -> Result<Json<ModelInfoBody>, ApiError> {
    let info = inference.model_info(&model_name, query.load).await?;
    debug!(model = %model_name, versions = info.versions.len(), "described model");
    Ok(Json(info.into()))
}

async fn model_metadata(
    State(inference): State<AppState>,
    Path(model_name): Path<String>,
) -> Result<Json<ModelMetadataBody>, ApiError> {
    metadata(inference, ModelRef::latest(model_name)).await
}

async fn model_version_metadata(
    State(inference): State<AppState>,
    Path((model_name, model_version)): Path<(String, String)>,
) -> Result<Json<ModelMetadataBody>, ApiError> {
    metadata(inference, ModelRef::new(model_name, Some(model_version))).await
}

async fn metadata(
    inference: AppState,
    model: ModelRef,
) -> Result<Json<ModelMetadataBody>, ApiError> {
    Ok(Json(inference.metadata(&model).await?.into()))
}

async fn model_infer(
    State(inference): State<AppState>,
    Path(model_name): Path<String>,
    body: Result<Json<InferenceRequestBody>, axum::extract::rejection::JsonRejection>,
) -> Result<Json<InferenceResponseBody>, ApiError> {
    infer(inference, ModelRef::latest(model_name), body).await
}

async fn model_version_infer(
    State(inference): State<AppState>,
    Path((model_name, model_version)): Path<(String, String)>,
    body: Result<Json<InferenceRequestBody>, axum::extract::rejection::JsonRejection>,
) -> Result<Json<InferenceResponseBody>, ApiError> {
    infer(
        inference,
        ModelRef::new(model_name, Some(model_version)),
        body,
    )
    .await
}

async fn infer(
    inference: AppState,
    model: ModelRef,
    body: Result<Json<InferenceRequestBody>, axum::extract::rejection::JsonRejection>,
) -> Result<Json<InferenceResponseBody>, ApiError> {
    // A malformed body is the client's error, and must be reported in the
    // protocol's error object rather than axum's own plain-text rejection.
    let Json(body) = body.map_err(|e| {
        ApiError(InferError::InvalidRequest(format!(
            "could not read the request body: {}",
            e.body_text()
        )))
    })?;

    debug!(%model, inputs = body.inputs.len(), "REST inference request");
    let request = to_domain(model, body)?;
    let response = inference.infer(request).await?;
    Ok(Json(json::response_to_json(response)?))
}

/// Builds a domain request from the JSON body.
fn to_domain(model: ModelRef, body: InferenceRequestBody) -> Result<InferRequest, ApiError> {
    let mut inputs = Vec::with_capacity(body.inputs.len());
    for input in &body.inputs {
        let datatype = DataType::from_str(&input.datatype)?;
        let encoding = tensor_json::BytesEncoding::from_parameters(input.parameters.as_ref())?;
        let tensor = Tensor::new(
            &input.name,
            input.shape.clone(),
            tensor_json::decode(&input.name, datatype, &input.data, encoding)?,
        );
        tensor.validate()?;
        inputs.push(tensor);
    }

    Ok(InferRequest {
        model,
        id: body.id,
        parameters: parameters_from_json(body.parameters.as_ref())?,
        inputs,
        requested_outputs: body
            .outputs
            .unwrap_or_default()
            .into_iter()
            .map(|output| {
                Ok(RequestedOutput {
                    name: output.name,
                    parameters: parameters_from_json(output.parameters.as_ref())?,
                })
            })
            .collect::<Result<Vec<_>, ApiError>>()?,
    })
}

/// Reads JSON parameters into the domain's value kinds.
///
/// The gRPC form of the protocol allows a parameter to be a boolean, an int64
/// or a string, so one representation serves both transports and a
/// non-integral JSON number is refused rather than silently rewritten.
fn parameters_from_json(parameters: Option<&JsonParameters>) -> Result<Parameters, ApiError> {
    let mut out = Parameters::new();
    for (name, value) in parameters.into_iter().flatten() {
        let parameter = match value {
            Value::Bool(v) => Parameter::Bool(*v),
            Value::String(v) => Parameter::String(v.clone()),
            Value::Number(v) => match v.as_i64() {
                Some(v) => Parameter::Int64(v),
                None => {
                    return Err(ApiError(InferError::InvalidRequest(format!(
                        "parameter '{name}' is {v}; parameters may be a boolean, \
                         an integer or a string"
                    ))))
                }
            },
            other => {
                return Err(ApiError(InferError::InvalidRequest(format!(
                    "parameter '{name}' is {other}; parameters may be a boolean, \
                     an integer or a string"
                ))))
            }
        };
        out.insert(name.clone(), parameter);
    }
    Ok(out)
}

/// Serves Prometheus metrics.
///
/// This is not part of the Predict Protocol, which is why it sits outside `/v2`
/// and is mounted only when a metrics source is supplied: a deployment that
/// does not want metrics exposed simply does not get the route.
pub fn metrics_router(metrics: Arc<dyn MetricsSource>) -> Router {
    Router::new()
        .route("/metrics", get(scrape))
        .with_state(metrics)
}

async fn scrape(State(metrics): State<Arc<dyn MetricsSource>>) -> impl IntoResponse {
    (
        [(
            axum::http::header::CONTENT_TYPE,
            // The version Prometheus itself advertises for text exposition.
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        metrics.prometheus_text().await,
    )
}

/// Anything outside the protocol's URLs.
pub async fn not_found() -> impl IntoResponse {
    (
        StatusCode::NOT_FOUND,
        Json(ErrorBody {
            error: "no such endpoint; this server implements the Predict Protocol v2 \
                    under /v2"
                .to_string(),
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn body(json: Value) -> InferenceRequestBody {
        serde_json::from_value(json).expect("valid body")
    }

    #[test]
    fn the_specifications_example_request_decodes() {
        let request = to_domain(
            ModelRef::latest("mymodel"),
            body(json!({
                "id": "42",
                "inputs": [
                    {"name": "input0", "shape": [2, 2], "datatype": "UINT32",
                     "data": [1, 2, 3, 4]},
                    {"name": "input1", "shape": [3], "datatype": "BOOL",
                     "data": [true, false, true]}
                ],
                "outputs": [{"name": "output0"}]
            })),
        )
        .map_err(|e| e.0)
        .unwrap();

        assert_eq!(request.id.as_deref(), Some("42"));
        assert_eq!(request.inputs.len(), 2);
        assert_eq!(request.inputs[0].datatype(), DataType::Uint32);
        assert_eq!(request.inputs[0].shape, vec![2, 2]);
        assert_eq!(request.inputs[1].datatype(), DataType::Bool);
        assert_eq!(request.requested_outputs[0].name, "output0");
    }

    #[test]
    fn an_unknown_datatype_is_a_client_error() {
        let err = to_domain(
            ModelRef::latest("m"),
            body(json!({"inputs": [
                {"name": "a", "shape": [1], "datatype": "MAGIC", "data": [1]}
            ]})),
        )
        .map_err(|e| e.0)
        .unwrap_err();
        assert!(err.is_client_error(), "{err}");
    }

    #[test]
    fn data_that_disagrees_with_the_shape_is_rejected() {
        let err = to_domain(
            ModelRef::latest("m"),
            body(json!({"inputs": [
                {"name": "a", "shape": [2, 2], "datatype": "FP32", "data": [1.0]}
            ]})),
        )
        .map_err(|e| e.0)
        .unwrap_err();
        assert!(err.to_string().contains("4 elements"), "{err}");
    }

    #[test]
    fn parameters_accept_the_three_protocol_value_kinds() {
        let request = to_domain(
            ModelRef::latest("m"),
            body(json!({
                "parameters": {"flag": true, "count": 3, "tag": "x"},
                "inputs": [{"name": "a", "shape": [1], "datatype": "FP32", "data": [1.0]}]
            })),
        )
        .map_err(|e| e.0)
        .unwrap();

        assert_eq!(request.parameters["flag"], Parameter::Bool(true));
        assert_eq!(request.parameters["count"], Parameter::Int64(3));
        assert_eq!(request.parameters["tag"], Parameter::String("x".into()));
    }

    #[test]
    fn a_parameter_the_protocol_cannot_carry_is_refused() {
        for value in [json!(1.5), json!(["a"]), json!({"a": 1})] {
            let err = to_domain(
                ModelRef::latest("m"),
                body(json!({
                    "parameters": {"p": value},
                    "inputs": [{"name": "a", "shape": [1], "datatype": "FP32", "data": [1.0]}]
                })),
            )
            .map_err(|e| e.0)
            .unwrap_err();
            assert!(err.is_client_error(), "{err}");
        }
    }

    #[test]
    fn health_answers_with_a_status_and_no_body() {
        assert_eq!(health(true), StatusCode::OK);
        assert_eq!(health(false), StatusCode::BAD_REQUEST);
    }
}
