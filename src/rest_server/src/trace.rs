//! Per-request tracing.
//!
//! Every request is given an id and a span, and both ends of it are logged:
//! one line when it arrives and one when it is answered, carrying the status
//! and how long it took. Everything the server logs while handling the request
//! -- version resolution, a model load, queueing, the backend call -- is
//! nested inside that span, so a single request can be followed through a log
//! that interleaves hundreds of them.
//!
//! The id is taken from `X-Request-Id` when the caller sent one, so a trace
//! can be followed from a gateway's log into this one, and is echoed on the
//! response either way, so a caller that did not send one can still quote the
//! id of a request that went wrong.
//!
//! The arrival line is at debug and the completion line at debug for a
//! success, warn for a server error. At the default `info` filter a healthy
//! server therefore says nothing per request -- a server answering 20k
//! requests a second cannot log two lines each -- and `GALEMIND_LOG=debug`
//! turns the flow on.

use axum::extract::Request;
use axum::http::{HeaderName, HeaderValue};
use axum::middleware::Next;
use axum::response::Response;
use galemind_core::request_id;
use std::time::Instant;
use tracing::{debug, field::Empty, warn, Instrument};

/// The header a request id is read from and echoed in.
pub const REQUEST_ID_HEADER: HeaderName = HeaderName::from_static("x-request-id");

/// Extension holding the id of the request being handled.
///
/// Handlers read it to name the request in their own log lines; it is a
/// newtype so that it cannot collide with another `String` extension.
#[derive(Debug, Clone)]
pub struct RequestId(pub String);

/// Gives every request an id, a span and a pair of log lines.
pub async fn trace_requests(mut request: Request, next: Next) -> Response {
    let id = request_id::request_id_or_new(
        request
            .headers()
            .get(&REQUEST_ID_HEADER)
            .and_then(|value| value.to_str().ok()),
    );
    let method = request.method().clone();
    // The path, without the query string: a path is one of the protocol's
    // eight URLs and makes a usable log field, where a query string is the
    // caller's to put anything in.
    let path = request.uri().path().to_string();

    // `status` is declared empty and recorded on the way out, so that the
    // span -- and so every line logged inside it -- carries the outcome.
    let span = tracing::info_span!(
        "request",
        id = %id,
        method = %method,
        path = %path,
        status = Empty,
    );
    request.extensions_mut().insert(RequestId(id.clone()));

    let at = Instant::now();
    let mut response = async move {
        debug!("request received");
        let response = next.run(request).await;

        let status = response.status();
        let elapsed_us = at.elapsed().as_micros();
        tracing::Span::current().record("status", status.as_u16());
        if status.is_server_error() {
            warn!(elapsed_us, "request failed");
        } else {
            debug!(elapsed_us, "request answered");
        }
        response
    }
    .instrument(span)
    .await;

    // Echoed so that a caller can quote the id of a request that went wrong,
    // including one that did not send an id of its own. The id is checked
    // before use, so it is always a valid header value.
    if let Ok(value) = HeaderValue::from_str(&id) {
        response.headers_mut().insert(REQUEST_ID_HEADER, value);
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::routing::get;
    use axum::Router;
    use tower::ServiceExt;

    /// A handler that reports the id the middleware gave the request, so a
    /// test can see what a handler would log.
    async fn echo_id(axum::Extension(RequestId(id)): axum::Extension<RequestId>) -> String {
        id
    }

    fn app() -> Router {
        Router::new()
            .route("/echo", get(echo_id))
            .layer(axum::middleware::from_fn(trace_requests))
    }

    async fn call(request: Request<Body>) -> (Option<String>, String) {
        let response = app().oneshot(request).await.expect("the app answers");
        let header = response
            .headers()
            .get(&REQUEST_ID_HEADER)
            .map(|value| value.to_str().expect("an ASCII id").to_string());
        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .expect("a body");
        (header, String::from_utf8(body.to_vec()).expect("UTF-8"))
    }

    #[tokio::test]
    async fn a_callers_id_is_used_and_echoed_so_one_id_spans_both_logs() {
        let (header, body) = call(
            Request::get("/echo")
                .header("x-request-id", "gateway-42")
                .body(Body::empty())
                .unwrap(),
        )
        .await;

        assert_eq!(header.as_deref(), Some("gateway-42"));
        assert_eq!(body, "gateway-42", "the handler sees the same id");
    }

    #[tokio::test]
    async fn a_request_without_an_id_is_given_one() {
        let (header, body) = call(Request::get("/echo").body(Body::empty()).unwrap()).await;

        let id = header.expect("an id is echoed");
        assert!(!id.is_empty());
        assert_eq!(body, id);
    }

    /// An id reaches a response header and the log, so one that could forge a
    /// header or a log line is replaced rather than passed through.
    #[tokio::test]
    async fn an_unusable_caller_id_is_replaced() {
        let (header, _) = call(
            Request::get("/echo")
                .header(
                    "x-request-id",
                    &"x".repeat(request_id::MAX_REQUEST_ID_LEN + 1),
                )
                .body(Body::empty())
                .unwrap(),
        )
        .await;

        let id = header.expect("an id is echoed");
        assert!(id.len() <= request_id::MAX_REQUEST_ID_LEN, "{id}");
    }

    #[tokio::test]
    async fn two_requests_get_two_ids() {
        let (first, _) = call(Request::get("/echo").body(Body::empty()).unwrap()).await;
        let (second, _) = call(Request::get("/echo").body(Body::empty()).unwrap()).await;
        assert_ne!(first, second);
    }
}
