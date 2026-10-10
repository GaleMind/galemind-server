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
mod metrics;
mod rows;

pub use grpc::GrpcClient;
pub use http::HttpClient;
pub use metrics::{scrape, Sample, Scrape};
pub use rows::{parse_row, row_width, rows_to_tensor, sole_input, synthesize_rows};

use async_trait::async_trait;
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

    /// Which transport this client speaks, for logging and reports.
    fn transport(&self) -> &'static str;
}

/// Which transport to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transport {
    Http,
    Grpc,
}

impl std::fmt::Display for Transport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Http => "http",
            Self::Grpc => "grpc",
        })
    }
}

/// Connects a client of the given transport to `endpoint`.
pub async fn connect(
    transport: Transport,
    endpoint: &str,
) -> Result<Box<dyn PredictV2Client>, InferError> {
    Ok(match transport {
        Transport::Http => Box::new(HttpClient::new(endpoint)?),
        Transport::Grpc => Box::new(GrpcClient::connect(endpoint).await?),
    })
}
