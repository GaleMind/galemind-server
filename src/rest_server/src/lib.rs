//! The Predict Protocol v2 HTTP/REST transport.
//!
//! The protocol's URLs and nothing else: no chat endpoints, no
//! protocol-selection header, no vendor extensions. Handlers decode a body
//! with [`galemind_core::json`], call [`galemind_core::Inference`] and encode
//! the answer.

#![forbid(unsafe_code)]

pub mod auth;
mod error;
mod routes;
mod trace;

pub use auth::{AuthError, Authenticator, Identity, NoAuth, StaticTokenAuth};
pub use error::status_for;
pub use routes::{metrics_router, router as predict_v2_router};
pub use trace::{RequestId, REQUEST_ID_HEADER};

use axum::Router;
use galemind_core::{Inference, MetricsSource};
use galemind_service::{bind_with_backlog, DEFAULT_LISTEN_BACKLOG};
use std::net::SocketAddr;
use std::sync::Arc;
use tracing::info;

/// Builds the application, with per-request tracing and a protocol-shaped 404.
///
/// `metrics` is optional: without it the server exposes the protocol and
/// nothing else, so metrics reporting can be turned off entirely.
///
/// The tracing layer is outermost, so a request that is rejected before it
/// reaches a handler -- an unknown URL, a body the extractor refuses -- is
/// logged and carries a request id too.
pub fn app(inference: Arc<dyn Inference>, metrics: Option<Arc<dyn MetricsSource>>) -> Router {
    app_with_auth(inference, metrics, Arc::new(NoAuth))
}

/// Builds the application with an identity check in front of it.
///
/// Layer order is from the outside in: tracing first, so that a request
/// refused by the identity check is still logged with an id; then
/// authentication, so that no handler runs for a request that was refused.
pub fn app_with_auth(
    inference: Arc<dyn Inference>,
    metrics: Option<Arc<dyn MetricsSource>>,
    authenticator: Arc<dyn Authenticator>,
) -> Router {
    let mut app = predict_v2_router(inference);
    if let Some(metrics) = metrics {
        app = app.merge(metrics_router(metrics));
    }
    app.fallback(routes::not_found)
        .layer(axum::middleware::from_fn_with_state(
            authenticator,
            auth::authenticate,
        ))
        .layer(axum::middleware::from_fn(trace::trace_requests))
}

/// Runs the REST server until `shutdown` resolves, without authentication.
pub async fn serve(
    address: SocketAddr,
    inference: Arc<dyn Inference>,
    metrics: Option<Arc<dyn MetricsSource>>,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    serve_with_auth(
        address,
        inference,
        metrics,
        Arc::new(NoAuth),
        DEFAULT_LISTEN_BACKLOG,
        shutdown,
    )
    .await
}

/// Runs the REST server until `shutdown` resolves.
pub async fn serve_with_auth(
    address: SocketAddr,
    inference: Arc<dyn Inference>,
    metrics: Option<Arc<dyn MetricsSource>>,
    authenticator: Arc<dyn Authenticator>,
    backlog: u32,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Not `TcpListener::bind`, whose backlog of 1024 is what a C10K burst
    // actually hits first; see `galemind_service::listener`.
    let listener = bind_with_backlog(address, backlog).await?;
    info!(
        address = %listener.local_addr()?,
        auth = authenticator.name(),
        backlog,
        "REST server listening"
    );

    axum::serve(listener, app_with_auth(inference, metrics, authenticator))
        .with_graceful_shutdown(shutdown)
        .await?;
    info!("REST server stopped");
    Ok(())
}
