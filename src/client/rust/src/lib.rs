//! A Predict Protocol v2 client.
//!
//! The same six calls over either transport, behind one trait, so a program
//! chooses HTTP or gRPC by configuration rather than by being written twice.
//! Requests and responses are the domain types from [`galemind_core`], and the
//! JSON encoding is the shared one in [`galemind_core::json`], so client and
//! server cannot drift apart.

#![forbid(unsafe_code)]

mod grpc;
mod http;
pub mod limits;
mod load;
mod metrics;
mod rows;

pub use grpc::GrpcClient;
pub use http::{HttpClient, HttpOptions};
pub use limits::{check_concurrency, check_sockets, file_limits, FileLimits};
pub use load::{run_load, LoadConfig, LoadReport, Percentiles, Summary, Tick};
pub use metrics::{scrape, Sample, Scrape};
pub use rows::{
    parse_row, row_width, rows_to_tensor, sole_input, synthesize_inputs, synthesize_rows,
};

use async_trait::async_trait;
use galemind_core::json::{ModelInfoBody, ModelListBody};
use galemind_core::{
    InferError, InferRequest, InferResponse, ModelMetadata, ModelRef, ServerMetadata,
};

/// The protocol's six calls.
#[async_trait]
pub trait PredictV2Client: Send + Sync {
    /// Whether the server can receive and answer requests.
    async fn server_live(&self) -> Result<bool, InferError>;

    /// Whether every model is ready for inference.
    async fn server_ready(&self) -> Result<bool, InferError>;

    /// Whether one model is ready for inference.
    async fn model_ready(&self, model: &ModelRef) -> Result<bool, InferError>;

    async fn server_metadata(&self) -> Result<ServerMetadata, InferError>;

    async fn model_metadata(&self, model: &ModelRef) -> Result<ModelMetadata, InferError>;

    async fn infer(&self, request: InferRequest) -> Result<InferResponse, InferError>;

    /// Every model the server can serve, from `/v2/models/list`.
    ///
    /// Not one of the protocol's six calls, so a transport that cannot make
    /// it says so rather than pretending: the gRPC service has no catalogue
    /// method, and inventing one by listing over HTTP from a gRPC client
    /// would hide which port answered. The reply is the JSON body as it
    /// stands, because the catalogue is this server's own shape rather than
    /// part of the protocol's domain.
    async fn model_list(&self) -> Result<ModelListBody, InferError> {
        Err(InferError::Unsupported(format!(
            "the {} transport has no model catalogue call; it is served over HTTP \
             at /v2/models/list",
            self.transport()
        )))
    }

    /// Every version of one model, from `/v2/models/info/{name}`.
    ///
    /// With `load`, the server loads each version so that all of them report
    /// a signature; see the endpoint's own documentation.
    async fn model_info(&self, _name: &str, _load: bool) -> Result<ModelInfoBody, InferError> {
        Err(InferError::Unsupported(format!(
            "the {} transport has no model info call; it is served over HTTP \
             at /v2/models/info/{{name}}",
            self.transport()
        )))
    }

    /// Which transport this client speaks, for logging and reports.
    fn transport(&self) -> &'static str;

    /// Sockets this client will hold open to keep `concurrency` requests in
    /// flight.
    ///
    /// The conservative default, one per request, is HTTP/1.1's answer: a
    /// request owns its connection until it answers. A multiplexed transport
    /// overrides it with the size of its connection pool, which is what makes
    /// the difference between a C10K run needing ten thousand descriptors and
    /// needing a hundred.
    fn sockets_for(&self, concurrency: usize) -> usize {
        concurrency.max(1)
    }
}

/// Which transport to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transport {
    /// HTTP/1.1: one request per connection at a time.
    Http,
    /// HTTP/2 with prior knowledge, on the same port: many requests per
    /// connection, as streams.
    ///
    /// The same URLs and the same bodies -- it is the framing underneath that
    /// changes -- so everything the HTTP client can do, this can do, on a
    /// fraction of the sockets.
    Http2,
    /// gRPC, which is HTTP/2 and so multiplexed by construction.
    Grpc,
}

impl std::fmt::Display for Transport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Http => "http",
            Self::Http2 => "http2",
            Self::Grpc => "grpc",
        })
    }
}

/// Connects a client of the given transport to `endpoint`.
///
/// Sized for occasional use: a handful of connections, which is what a
/// command-line call or an application making a request at a time needs. A
/// load run wants [`connect_for_concurrency`] instead.
pub async fn connect(
    transport: Transport,
    endpoint: &str,
) -> Result<Box<dyn PredictV2Client>, InferError> {
    connect_with(transport, endpoint, 1, None).await
}

/// Connects a client built to keep `concurrency` requests in flight.
///
/// Both transports need telling: HTTP needs a connection pool that can hold a
/// connection per worker, and gRPC needs more than one connection to spread
/// the streams across. Left at their defaults, each becomes the bottleneck
/// well before the server does, and a run then measures the client.
pub async fn connect_for_concurrency(
    transport: Transport,
    endpoint: &str,
    concurrency: usize,
) -> Result<Box<dyn PredictV2Client>, InferError> {
    connect_with(transport, endpoint, concurrency, None).await
}

/// What a client should be built for.
///
/// Three parameters rather than three overloads, because they are chosen
/// together: a token for a server that requires one, a concurrency the
/// client has to be able to carry, and -- where the transport multiplexes --
/// how many sockets it may spread that concurrency over.
#[derive(Debug, Clone, Default)]
pub struct ClientConfig {
    /// Requests the client must keep in flight at once.
    pub concurrency: usize,
    /// Sent on every request, for a server that requires one.
    pub bearer_token: Option<String>,
    /// The most sockets a multiplexed client may open.
    ///
    /// A ceiling on a count that otherwise follows the concurrency, not the
    /// count itself. `None` takes the deployment's:
    /// `GALEMIND_MAX_CONNECTIONS`, or 256 -- see
    /// [`limits::max_connections`].
    ///
    /// It has no effect under HTTP/1.1, where a request owns its connection
    /// until it answers, so the pool has to follow the concurrency or the
    /// surplus requests spend the run reconnecting.
    pub max_connections: Option<usize>,
}

impl ClientConfig {
    /// A client for `concurrency` requests at once, with everything else
    /// derived.
    pub fn for_concurrency(concurrency: usize) -> Self {
        Self {
            concurrency,
            ..Self::default()
        }
    }

    pub fn with_bearer_token(mut self, token: Option<&str>) -> Self {
        self.bearer_token = token.map(str::to_string);
        self
    }

    /// Caps the sockets a multiplexed client may open.
    pub fn with_max_connections(mut self, connections: Option<usize>) -> Self {
        self.max_connections = connections.map(|count| count.max(1));
        self
    }
}

/// Connects a client built to the given plan.
///
/// A token is sent on either transport -- as an `Authorization` header over
/// HTTP and as `authorization` metadata over gRPC, which is the same header
/// on the wire -- and the server checks both against one authenticator.
pub async fn connect_as(
    transport: Transport,
    endpoint: &str,
    config: &ClientConfig,
) -> Result<Box<dyn PredictV2Client>, InferError> {
    let concurrency = config.concurrency.max(1);
    Ok(match transport {
        Transport::Http | Transport::Http2 => {
            let mut options = if transport == Transport::Http2 {
                HttpOptions::multiplexed(
                    concurrency,
                    config
                        .max_connections
                        .unwrap_or_else(limits::max_connections),
                )
            } else {
                // Not capped: under HTTP/1.1 a pool below the concurrency is
                // not fewer sockets, it is the same sockets opened again and
                // again.
                HttpOptions::for_concurrency(concurrency)
            };
            if let Some(token) = &config.bearer_token {
                options = options.with_bearer_token(token.as_str());
            }
            Box::new(HttpClient::with_options(endpoint, options)?)
        }
        Transport::Grpc => {
            let client = GrpcClient::connect_under(
                endpoint,
                concurrency,
                config
                    .max_connections
                    .unwrap_or_else(limits::max_connections),
            )
            .await?;
            Box::new(match &config.bearer_token {
                Some(token) => client.with_bearer_token(token.as_str())?,
                None => client,
            })
        }
    })
}

/// Connects with a concurrency and an optional token, deriving the rest.
pub async fn connect_with(
    transport: Transport,
    endpoint: &str,
    concurrency: usize,
    bearer_token: Option<&str>,
) -> Result<Box<dyn PredictV2Client>, InferError> {
    connect_as(
        transport,
        endpoint,
        &ClientConfig::for_concurrency(concurrency).with_bearer_token(bearer_token),
    )
    .await
}
