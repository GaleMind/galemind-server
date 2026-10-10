//! The Predict Protocol v2 HTTP/REST transport.
//!
//! The protocol's URLs and nothing else: no chat endpoints, no
//! protocol-selection header, no vendor extensions. Handlers decode a body
//! with [`galemind_core::json`], call [`galemind_core::Inference`] and encode
//! the answer.

#![forbid(unsafe_code)]

mod error;
mod routes;

pub use error::status_for;
pub use routes::{metrics_router, router as predict_v2_router};

use axum::Router;
use galemind_core::{Inference, MetricsSource};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tower_http::trace::TraceLayer;
use tracing::info;

/// Builds the application, with request tracing and a protocol-shaped 404.
///
/// `metrics` is optional: without it the server exposes the protocol and
/// nothing else, so metrics reporting can be turned off entirely.
pub fn app(inference: Arc<dyn Inference>, metrics: Option<Arc<dyn MetricsSource>>) -> Router {
    let mut app = predict_v2_router(inference);
    if let Some(metrics) = metrics {
        app = app.merge(metrics_router(metrics));
    }
    app.fallback(routes::not_found)
        .layer(TraceLayer::new_for_http())
}

/// Runs the REST server until `shutdown` resolves.
pub async fn serve(
    address: SocketAddr,
    inference: Arc<dyn Inference>,
    metrics: Option<Arc<dyn MetricsSource>>,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let listener = TcpListener::bind(address).await?;
    info!(address = %listener.local_addr()?, "REST server listening");

    axum::serve(listener, app(inference, metrics))
        .with_graceful_shutdown(shutdown)
        .await?;
    info!("REST server stopped");
    Ok(())
}
