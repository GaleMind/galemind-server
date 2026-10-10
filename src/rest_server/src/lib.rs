//! The Predict Protocol v2 HTTP/REST transport.
//!
//! The protocol's URLs and JSON objects, and nothing else: no chat endpoints,
//! no protocol-selection header, no vendor extensions. Handlers decode a body,
//! call [`galemind_core::Inference`] and encode the answer.

#![forbid(unsafe_code)]

mod error;
mod routes;
mod tensor_json;
mod wire;

pub use error::status_for;
pub use routes::router as predict_v2_router;
pub use tensor_json::{BytesEncoding, BINARY_ENCODING_PARAMETER};
pub use wire::{
    ErrorBody, InferenceRequestBody, InferenceResponseBody, MetadataTensor, ModelMetadataBody,
    RequestInput, RequestOutput, ResponseOutput, ServerMetadata,
};

use axum::Router;
use galemind_core::Inference;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tower_http::trace::TraceLayer;
use tracing::info;

/// Builds the application, with request tracing and a protocol-shaped 404.
pub fn app(inference: Arc<dyn Inference>) -> Router {
    predict_v2_router(inference)
        .fallback(routes::not_found)
        .layer(TraceLayer::new_for_http())
}

/// Runs the REST server until `shutdown` resolves.
pub async fn serve(
    address: SocketAddr,
    inference: Arc<dyn Inference>,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let listener = TcpListener::bind(address).await?;
    info!(address = %listener.local_addr()?, "REST server listening");

    axum::serve(listener, app(inference))
        .with_graceful_shutdown(shutdown)
        .await?;
    info!("REST server stopped");
    Ok(())
}
