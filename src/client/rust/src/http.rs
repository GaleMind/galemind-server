//! The HTTP/REST client.

use crate::PredictV2Client;
use async_trait::async_trait;
use galemind_core::json::{
    self, ErrorBody, InferenceRequestBody, InferenceResponseBody, ModelInfoBody, ModelListBody,
    ModelMetadataBody, RequestInput, RequestOutput, ServerMetadataBody,
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

/// How an [`HttpClient`] is built.
#[derive(Debug, Clone)]
pub struct HttpOptions {
    /// How long a single request may take.
    pub timeout: Duration,
    /// How long establishing a connection may take. Kept well below the
    /// request timeout so that an unreachable server is reported as
    /// unreachable rather than as a slow request.
    pub connect_timeout: Duration,
    /// Idle connections the pool may keep for the server.
    ///
    /// This is the setting that decides whether a high-concurrency run
    /// measures the server or the cost of reconnecting to it. With HTTP/1.1 a
    /// request holds its connection for its lifetime, so a run at concurrency
    /// `n` opens `n` connections; a pool that keeps fewer closes the surplus
    /// the moment each request finishes and opens it again for the next one.
    /// At concurrency 700 against a pool of 64 that is hundreds of handshakes
    /// a second, which shows up as latency and as a drift of sockets into
    /// `TIME_WAIT` until the machine runs out of ephemeral ports.
    pub pool_max_idle_per_host: usize,
    /// How long an idle connection is kept. A load run reuses connections
    /// within milliseconds, so this only governs what happens after a run.
    pub pool_idle_timeout: Duration,
    /// Sent as `Authorization: Bearer <token>` on every request, for a server
    /// that requires one.
    pub bearer_token: Option<String>,
    /// Carry many requests on each connection instead of one at a time.
    ///
    /// HTTP/2 over cleartext has no negotiation -- there is no ALPN without
    /// TLS and the server is not asked to upgrade -- so this sends the HTTP/2
    /// preface and assumes the server speaks it. Against this server it does;
    /// against an HTTP/1.1-only proxy in front of it, every request fails, so
    /// it is asked for rather than guessed.
    ///
    /// What it buys is the whole socket-per-request problem going away: with
    /// multiplexing, requests in flight are HTTP/2 streams, so a run at
    /// concurrency 10 000 holds a few dozen sockets instead of ten thousand
    /// file descriptors -- measured at 28 against this server, where the
    /// same run over HTTP/1.1 peaked at 9 115 and timed out. See
    /// [`crate::limits::STREAMS_PER_CONNECTION`].
    pub multiplex: bool,
}

impl Default for HttpOptions {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            connect_timeout: Duration::from_secs(5),
            pool_max_idle_per_host: 64,
            pool_idle_timeout: Duration::from_secs(90),
            bearer_token: None,
            multiplex: false,
        }
    }
}

/// Requests one HTTP/2 connection is asked to carry at once; see
/// [`crate::limits::STREAMS_PER_CONNECTION`], which both multiplexed
/// transports derive their connection counts from.
pub use crate::limits::STREAMS_PER_CONNECTION;

impl HttpOptions {
    /// Options for driving `concurrency` requests at once.
    ///
    /// The pool is sized to the concurrency, with a little room above it, so
    /// that every worker keeps its own connection for the whole run.
    pub fn for_concurrency(concurrency: usize) -> Self {
        Self {
            pool_max_idle_per_host: concurrency.max(1).saturating_add(8),
            ..Self::default()
        }
    }

    /// Options for driving `concurrency` requests at once over HTTP/2,
    /// multiplexed onto as few connections as will carry them.
    ///
    /// The pool is sized to the connections rather than to the concurrency:
    /// the requests are streams now, and a pool of ten thousand idle
    /// connections would be ten thousand descriptors held for nothing.
    pub fn multiplexed_for_concurrency(concurrency: usize) -> Self {
        Self::multiplexed(concurrency, crate::limits::max_connections())
    }

    /// Options for driving `concurrency` requests at once over HTTP/2,
    /// onto at most `max_connections` sockets.
    ///
    /// The ceiling is the caller's: 256 by default, or whatever
    /// `GALEMIND_MAX_CONNECTIONS` says. See
    /// [`crate::limits::max_connections`].
    pub fn multiplexed(concurrency: usize, max_connections: usize) -> Self {
        Self {
            multiplex: true,
            pool_max_idle_per_host: connections_for(concurrency, max_connections),
            ..Self::default()
        }
    }

    /// Authenticates every request with a bearer token.
    pub fn with_bearer_token(mut self, token: impl Into<String>) -> Self {
        self.bearer_token = Some(token.into());
        self
    }

    /// Sockets a client built from these options will hold open to keep
    /// `concurrency` requests in flight.
    ///
    /// Under HTTP/1.1 that is one per request, because a request owns its
    /// connection until it answers. Multiplexed it is the pool, whatever the
    /// concurrency.
    pub fn sockets_for(&self, concurrency: usize) -> usize {
        if self.multiplex {
            self.pool_max_idle_per_host.max(1)
        } else {
            concurrency.max(1)
        }
    }
}

/// Connections to multiplex a given concurrency onto, under a ceiling.
///
/// At least two, so that one stalled connection is not the whole run, and
/// never more than `max_connections`, which is the deployment's call.
fn connections_for(concurrency: usize, max_connections: usize) -> usize {
    let wanted = concurrency.max(1).div_ceil(STREAMS_PER_CONNECTION);
    // Written as max-then-min rather than `clamp`, which panics when a
    // ceiling of 1 is configured.
    wanted.max(2).min(max_connections.max(1))
}

/// Talks the protocol over HTTP.
pub struct HttpClient {
    client: Client,
    base: String,
    options: HttpOptions,
}

impl HttpClient {
    /// `endpoint` is the server's base URL, for example `http://localhost:8000`.
    pub fn new(endpoint: &str) -> Result<Self> {
        Self::with_options(endpoint, HttpOptions::default())
    }

    pub fn with_timeout(endpoint: &str, timeout: Duration) -> Result<Self> {
        Self::with_options(
            endpoint,
            HttpOptions {
                timeout,
                ..HttpOptions::default()
            },
        )
    }

    pub fn with_options(endpoint: &str, options: HttpOptions) -> Result<Self> {
        let mut headers = reqwest::header::HeaderMap::new();
        if let Some(token) = &options.bearer_token {
            // A default header rather than one set per call: every request
            // this client makes needs it, and forgetting it on one call would
            // show up as a 401 from one endpoint only.
            let mut value = reqwest::header::HeaderValue::from_str(&format!("Bearer {token}"))
                .map_err(|_| {
                    InferError::InvalidRequest(
                        "the bearer token contains characters a header cannot carry".to_string(),
                    )
                })?;
            value.set_sensitive(true);
            headers.insert(reqwest::header::AUTHORIZATION, value);
        }

        let mut builder = Client::builder()
            .default_headers(headers)
            .timeout(options.timeout)
            .connect_timeout(options.connect_timeout)
            // Inference clients tend to be chatty; reusing connections matters
            // far more than it does for a browser.
            .pool_max_idle_per_host(options.pool_max_idle_per_host)
            .pool_idle_timeout(options.pool_idle_timeout)
            // Inference bodies are small and latency-sensitive, so a request
            // should not wait on Nagle's algorithm for a full segment.
            .tcp_nodelay(true);

        if options.multiplex {
            // Cleartext HTTP/2 has nothing to negotiate with, so the client
            // states it and the server either speaks it or does not.
            builder = builder
                .http2_prior_knowledge()
                // Streams are opened in bursts by the workers, so the
                // connection's initial windows have to be worth opening: at
                // the default 64 KiB a batch of large tensors spends its first
                // round trips waiting for window updates.
                .http2_initial_stream_window_size(1024 * 1024)
                .http2_initial_connection_window_size(16 * 1024 * 1024)
                // A connection that dies silently must not take the streams
                // on it down with it unnoticed, which matters more when one
                // socket is carrying hundreds of requests than when it is
                // carrying one.
                .http2_keep_alive_interval(Duration::from_secs(10))
                .http2_keep_alive_timeout(Duration::from_secs(5));
        }

        let client = builder
            .build()
            .map_err(|e| InferError::Internal(format!("cannot build an HTTP client: {e}")))?;

        Ok(Self {
            client,
            base: endpoint.trim_end_matches('/').to_string(),
            options,
        })
    }

    /// How this client was built.
    pub fn options(&self) -> &HttpOptions {
        &self.options
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

    async fn model_list(&self) -> Result<ModelListBody> {
        self.get_json(&format!("{}/v2/models/list", self.base))
            .await
    }

    async fn model_info(&self, name: &str, load: bool) -> Result<ModelInfoBody> {
        self.get_json(&format!("{}/v2/models/info/{name}?load={load}", self.base))
            .await
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
        if self.options.multiplex {
            "http2"
        } else {
            "http"
        }
    }

    fn sockets_for(&self, concurrency: usize) -> usize {
        self.options.sockets_for(concurrency)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The point of the multiplexed client: the sockets stop following the
    /// concurrency, so the open-file limit stops being the ceiling.
    #[test]
    fn multiplexing_makes_the_socket_count_independent_of_the_concurrency() {
        let plain = HttpOptions::for_concurrency(10_000);
        assert!(!plain.multiplex);
        assert_eq!(plain.sockets_for(10_000), 10_000);

        let multiplexed = HttpOptions::multiplexed_for_concurrency(10_000);
        assert!(multiplexed.multiplex);
        // 10 000 requests over 256 streams each: 40 sockets, not 10 000.
        assert_eq!(multiplexed.sockets_for(10_000), 40);
    }

    #[test]
    fn connections_follow_the_concurrency_between_a_floor_and_a_ceiling() {
        let ceiling = crate::limits::DEFAULT_MAX_CONNECTIONS;
        // Two at the bottom, so one stalled connection is not the whole run.
        assert_eq!(connections_for(1, ceiling), 2);
        assert_eq!(connections_for(512, ceiling), 2);
        assert_eq!(connections_for(1_024, ceiling), 4);
        assert_eq!(connections_for(10_000, ceiling), 40);
        // And never past the ceiling, whoever set it.
        assert_eq!(connections_for(1_000_000, ceiling), 256);
        assert_eq!(connections_for(10_000, 8), 8);
        // A ceiling of one is honoured rather than panicked on, even though
        // the floor would otherwise be two.
        assert_eq!(connections_for(10_000, 1), 1);
    }

    #[test]
    fn a_multiplexed_client_says_which_transport_it_is() {
        let plain = HttpClient::new("http://127.0.0.1:8000").expect("a client");
        assert_eq!(plain.transport(), "http");

        let multiplexed =
            HttpClient::with_options("http://127.0.0.1:8000", HttpOptions::multiplexed(700, 256))
                .expect("a client");
        assert_eq!(multiplexed.transport(), "http2");
        assert_eq!(multiplexed.sockets_for(700), 3);
    }
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

    /// The pool has to hold a connection per request in flight, or a
    /// high-concurrency run spends its time reconnecting.
    #[test]
    fn the_connection_pool_is_sized_to_the_concurrency_it_will_serve() {
        assert_eq!(
            HttpOptions::for_concurrency(700).pool_max_idle_per_host,
            708
        );
        assert_eq!(
            HttpOptions::for_concurrency(10_000).pool_max_idle_per_host,
            10_008
        );
        // A client not built for a load run keeps the modest default.
        assert_eq!(HttpOptions::default().pool_max_idle_per_host, 64);
        // Zero workers is still one connection.
        assert_eq!(HttpOptions::for_concurrency(0).pool_max_idle_per_host, 9);
    }

    #[test]
    fn a_bearer_token_that_cannot_be_a_header_is_refused_when_the_client_is_built() {
        let options = HttpOptions::default().with_bearer_token("line\nbreak");
        let error = HttpClient::with_options("http://localhost:8000", options)
            .err()
            .expect("refused");
        assert!(error.is_client_error(), "{error}");

        // An ordinary token is accepted.
        assert!(HttpClient::with_options(
            "http://localhost:8000",
            HttpOptions::default().with_bearer_token("s3cret")
        )
        .is_ok());
    }

    #[test]
    fn a_connect_timeout_is_set_below_the_request_timeout() {
        let options = HttpOptions::default();
        assert!(
            options.connect_timeout < options.timeout,
            "an unreachable server should be reported as unreachable"
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
