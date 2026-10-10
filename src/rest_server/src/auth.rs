//! The HTTP side of authentication.
//!
//! The decision itself -- who a token belongs to and whether it is accepted
//! -- is [`galemind_core::auth`], which both transports share. What is here
//! is the part that is HTTP's: one axum middleware that parses the
//! `Authorization` header, answers a failure with the protocol's error object
//! and a `WWW-Authenticate` challenge rather than an empty 401, and puts the
//! caller's [`Identity`] in the request's extensions for handlers to read.
//!
//! Health and metrics are exempt, as they are on the gRPC port; see
//! [`galemind_core::auth::is_open_http_path`] for which URLs and why.
//!
//! To add a provider, implement [`Authenticator`] and pass it to
//! [`crate::app_with_auth`]; nothing here changes.

use axum::extract::{Request, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use galemind_core::auth::is_open_http_path;
use galemind_core::json::ErrorBody;
use std::sync::Arc;
use tracing::{debug, warn};

// Re-exported so that `rest_server::auth` remains the one import a binary
// needs to configure the HTTP port, wherever the trait itself lives.
pub use galemind_core::auth::{AuthError, Authenticator, Identity, NoAuth, StaticTokenAuth};

/// Authenticates a request, or refuses it.
///
/// Mounted with [`axum::middleware::from_fn_with_state`]; see
/// [`crate::app_with_auth`].
pub async fn authenticate(
    State(auth): State<Arc<dyn Authenticator>>,
    mut request: Request,
    next: Next,
) -> Response {
    if is_open_http_path(request.uri().path()) {
        return next.run(request).await;
    }

    let header = request
        .headers()
        .get(header::AUTHORIZATION)
        .map(|value| value.to_str().map(str::to_string));

    let token = match header {
        Some(Ok(header)) => match galemind_core::auth::bearer_token(&header) {
            Ok(token) => Some(token.to_string()),
            Err(e) => return refuse(e),
        },
        // A header that is not ASCII cannot be a bearer token.
        Some(Err(_)) => {
            return refuse(AuthError::Malformed(
                "the Authorization header is not text".to_string(),
            ))
        }
        None if auth.allows_anonymous() => None,
        None => return refuse(AuthError::Missing),
    };

    match auth.authenticate(token.as_deref()).await {
        Ok(identity) => {
            if !identity.is_anonymous() {
                debug!(subject = %identity.subject, "authenticated");
            }
            request.extensions_mut().insert(identity);
            next.run(request).await
        }
        Err(e) => refuse(e),
    }
}

/// Logs a refusal with its reason and answers with the protocol's error
/// object. The reason stays in the log; the caller gets the generic message.
fn refuse(error: AuthError) -> Response {
    warn!(
        status = error.http_status(),
        reason = error.detail(),
        "refused an unauthenticated request"
    );

    let status =
        StatusCode::from_u16(error.http_status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let mut response = (
        status,
        Json(ErrorBody {
            error: error.message().to_string(),
        }),
    )
        .into_response();

    // A 401 without this header tells a client it was refused but not how to
    // try again; RFC 9110 requires it. A 403 is not a challenge: the
    // credential was understood and does not reach far enough, so inviting a
    // retry would be a loop.
    if status == StatusCode::UNAUTHORIZED {
        response
            .headers_mut()
            .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use axum::body::Body;
    use axum::routing::get;
    use axum::Router;
    use tower::ServiceExt;

    /// Reports the identity the middleware established, so a test sees what a
    /// handler would.
    async fn whoami(identity: Option<axum::Extension<Identity>>) -> String {
        match identity {
            Some(axum::Extension(identity)) => identity.subject,
            None => "no identity".to_string(),
        }
    }

    fn app(auth: Arc<dyn Authenticator>) -> Router {
        Router::new()
            .route("/v2/models/list", get(whoami))
            .route("/v2/health/live", get(|| async { "live" }))
            .route("/v2/health/ready", get(|| async { "ready" }))
            .route("/v2/models/{model}/ready", get(|| async { "ready" }))
            .route(
                "/v2/models/{model}/versions/{version}/ready",
                get(|| async { "ready" }),
            )
            .route("/metrics", get(|| async { "# metrics" }))
            .layer(axum::middleware::from_fn_with_state(auth, authenticate))
    }

    async fn get_with(
        auth: Arc<dyn Authenticator>,
        path: &str,
        authorization: Option<&str>,
    ) -> (StatusCode, Option<String>, String) {
        let mut request = Request::get(path);
        if let Some(value) = authorization {
            request = request.header(header::AUTHORIZATION, value);
        }
        let response = app(auth)
            .oneshot(request.body(Body::empty()).unwrap())
            .await
            .expect("the app answers");

        let status = response.status();
        let challenge = response
            .headers()
            .get(header::WWW_AUTHENTICATE)
            .map(|value| value.to_str().unwrap().to_string());
        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .expect("a body");
        (
            status,
            challenge,
            String::from_utf8(body.to_vec()).expect("UTF-8"),
        )
    }

    fn no_auth() -> Arc<dyn Authenticator> {
        Arc::new(NoAuth)
    }

    fn static_auth() -> Arc<dyn Authenticator> {
        Arc::new(StaticTokenAuth::new("s3cret"))
    }

    #[tokio::test]
    async fn the_default_lets_every_request_through_as_anonymous() {
        let (status, _, body) = get_with(no_auth(), "/v2/models/list", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, "anonymous");
    }

    /// A token sent to a server that checks nothing must not be reported as an
    /// identity: an unauthenticated server would then look authenticated.
    #[tokio::test]
    async fn an_unchecked_token_does_not_become_an_identity() {
        let (status, _, body) =
            get_with(no_auth(), "/v2/models/list", Some("Bearer whatever")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, "anonymous");
    }

    #[tokio::test]
    async fn a_valid_token_names_the_caller_for_the_handler() {
        let (status, _, body) =
            get_with(static_auth(), "/v2/models/list", Some("Bearer s3cret")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, "static-token");
    }

    #[tokio::test]
    async fn the_bearer_scheme_is_matched_case_insensitively() {
        let (status, _, _) =
            get_with(static_auth(), "/v2/models/list", Some("bearer s3cret")).await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn a_missing_token_is_refused_with_a_challenge_and_the_protocols_error() {
        let (status, challenge, body) = get_with(static_auth(), "/v2/models/list", None).await;

        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(challenge.as_deref(), Some("Bearer"));
        let error: ErrorBody = serde_json::from_str(&body).expect("the protocol's error object");
        assert!(error.error.contains("Bearer"), "{}", error.error);
    }

    #[tokio::test]
    async fn a_wrong_token_is_refused_without_saying_why() {
        let (status, _, body) =
            get_with(static_auth(), "/v2/models/list", Some("Bearer wrong")).await;

        assert_eq!(status, StatusCode::UNAUTHORIZED);
        let error: ErrorBody = serde_json::from_str(&body).expect("the protocol's error object");
        assert!(
            !error.error.contains("match"),
            "the reason belongs in the log, not the response: {}",
            error.error
        );
    }

    #[tokio::test]
    async fn an_authorization_header_that_is_not_a_bearer_token_is_refused() {
        for header in ["Basic dXNlcjpwYXNz", "Bearer", "Bearer    ", "s3cret"] {
            let (status, _, _) = get_with(static_auth(), "/v2/models/list", Some(header)).await;
            assert_eq!(status, StatusCode::UNAUTHORIZED, "{header:?}");
        }
    }

    /// Switching authentication on must not take out the probes an
    /// orchestrator and Prometheus use, neither of which has a token to send.
    #[tokio::test]
    async fn every_health_check_and_the_metrics_scrape_stay_open() {
        for path in [
            "/v2/health/live",
            "/v2/health/ready",
            "/v2/models/thermal_cnn/ready",
            "/v2/models/thermal_cnn/versions/1/ready",
            "/metrics",
        ] {
            let (status, _, _) = get_with(static_auth(), path, None).await;
            assert_eq!(status, StatusCode::OK, "{path} must stay open");
        }
    }

    #[tokio::test]
    async fn a_forbidden_token_is_not_a_request_to_authenticate_again() {
        struct AlwaysForbidden;

        #[async_trait]
        impl Authenticator for AlwaysForbidden {
            fn name(&self) -> &'static str {
                "always-forbidden"
            }
            async fn authenticate(&self, _token: Option<&str>) -> Result<Identity, AuthError> {
                Err(AuthError::Forbidden("no roles".to_string()))
            }
        }

        let (status, challenge, _) = get_with(
            Arc::new(AlwaysForbidden),
            "/v2/models/list",
            Some("Bearer anything"),
        )
        .await;

        assert_eq!(status, StatusCode::FORBIDDEN);
        assert_eq!(challenge, None, "403 is not a challenge to retry");
    }
}
