//! The gRPC side of authentication.
//!
//! The decision itself -- who a token belongs to and whether it is accepted
//! -- is [`galemind_core::auth`], the same trait the HTTP port checks against,
//! so a deployment configures one authenticator and both ports enforce it.
//! What is here is the part that is gRPC's.
//!
//! # Why a tower layer and not an interceptor
//!
//! tonic's own seam is [`tonic::service::Interceptor`], which is a *synchronous*
//! `Request<()> -> Result<Request<()>, Status>`. That is enough for a shared
//! secret and not enough for anything that fetches: a real provider has to
//! await -- Keycloak's JWKS is an HTTP call, a cached key still expires, and
//! an introspection endpoint is a round trip. Doing that from a sync
//! interceptor means blocking a runtime thread per request, which is a way to
//! deadlock a server under load rather than a way to authenticate it.
//!
//! So the check is a [`tower::Layer`] in front of the router instead, where
//! the future is ours and an await is just an await. It sits at the same place
//! in the stack as the axum middleware on the other port, reads the same
//! `authorization` header from the HTTP/2 request tonic is carrying the call
//! on, and refuses with a trailers-only response built from a
//! [`tonic::Status`] -- which is what a gRPC client expects, rather than the
//! bare HTTP status an HTTP-shaped rejection would send.
//!
//! The identity is put in the request's extensions, where a service method
//! reads it with `request.extensions().get::<Identity>()`.

use galemind_core::auth::{is_open_grpc_method, AuthError, Authenticator};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tonic::body::Body;
use tonic::codegen::http::{Request, Response};
use tonic::{Code, Status};
use tower::{Layer, Service};
use tracing::{debug, warn};

// Re-exported so that `grpc_server::auth` is the one import a binary needs
// to configure the gRPC port, wherever the trait itself lives.
pub use galemind_core::auth::{Identity, NoAuth, StaticTokenAuth};

/// Puts an authentication check in front of a gRPC service.
///
/// Added with [`tonic::transport::Server::layer`], so it wraps every method of
/// every service on the port.
#[derive(Clone)]
pub struct AuthLayer {
    auth: Arc<dyn Authenticator>,
}

impl AuthLayer {
    pub fn new(auth: Arc<dyn Authenticator>) -> Self {
        Self { auth }
    }
}

impl<S> Layer<S> for AuthLayer {
    type Service = Authenticated<S>;

    fn layer(&self, inner: S) -> Self::Service {
        Authenticated {
            inner,
            auth: Arc::clone(&self.auth),
        }
    }
}

/// A gRPC service that checks the caller before dispatching to it.
#[derive(Clone)]
pub struct Authenticated<S> {
    inner: S,
    auth: Arc<dyn Authenticator>,
}

impl<S, B> Service<Request<B>> for Authenticated<S>
where
    S: Service<Request<B>, Response = Response<Body>> + Clone + Send + 'static,
    S::Future: Send + 'static,
    B: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut request: Request<B>) -> Self::Future {
        // `poll_ready` reserved capacity on `self.inner`, and it is that
        // clone which holds the reservation -- so the call has to be made on
        // it rather than on a fresh clone. The standard tower dance.
        let clone = self.inner.clone();
        let mut inner = std::mem::replace(&mut self.inner, clone);
        let auth = Arc::clone(&self.auth);

        Box::pin(async move {
            if is_open_grpc_method(request.uri().path()) {
                return inner.call(request).await;
            }

            let token = match token_of(&request) {
                Ok(token) => token,
                Err(e) => return Ok(refuse(e)),
            };

            let token = match token {
                None if !auth.allows_anonymous() => return Ok(refuse(AuthError::Missing)),
                other => other,
            };

            match auth.authenticate(token.as_deref()).await {
                Ok(identity) => {
                    if !identity.is_anonymous() {
                        debug!(subject = %identity.subject, "authenticated");
                    }
                    request.extensions_mut().insert(identity);
                    inner.call(request).await
                }
                Err(e) => Ok(refuse(e)),
            }
        })
    }
}

/// The bearer token on a call, if it carries one.
///
/// gRPC metadata is HTTP/2 headers, so this is the same `authorization`
/// header the REST port reads, and the same parser checks it.
fn token_of<B>(request: &Request<B>) -> Result<Option<String>, AuthError> {
    let Some(value) = request
        .headers()
        .get(tonic::codegen::http::header::AUTHORIZATION)
    else {
        return Ok(None);
    };

    let header = value
        .to_str()
        .map_err(|_| AuthError::Malformed("the Authorization header is not text".to_string()))?;

    galemind_core::auth::bearer_token(header).map(|token| Some(token.to_string()))
}

/// Refuses a call the way a gRPC client understands it.
///
/// A trailers-only response: HTTP 200 carrying `grpc-status` and
/// `grpc-message`, which is what [`Status::into_http`] builds. Answering with
/// a bare HTTP 401 instead would reach a gRPC client as a transport error
/// with no status code, so the one thing it needs to know -- that the
/// credential, not the network, was the problem -- would be the thing it
/// could not see.
fn refuse(error: AuthError) -> Response<Body> {
    let code = if error.is_forbidden() {
        Code::PermissionDenied
    } else {
        Code::Unauthenticated
    };

    warn!(
        code = ?code,
        reason = error.detail(),
        "refused an unauthenticated gRPC call"
    );

    // The caller gets the generic message; the reason stays in the log.
    Status::new(code, error.message()).into_http()
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::convert::Infallible;
    use tower::ServiceExt;

    const INFER: &str = "/inference.GRPCInferenceService/ModelInfer";
    const LIVE: &str = "/inference.GRPCInferenceService/ServerLive";
    const MODEL_READY: &str = "/inference.GRPCInferenceService/ModelReady";

    /// Stands in for the generated service, and reports what the layer passed
    /// it: the subject it would see in the extensions.
    #[derive(Clone)]
    struct Inner;

    impl Service<Request<Body>> for Inner {
        type Response = Response<Body>;
        type Error = Infallible;
        type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

        fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }

        fn call(&mut self, request: Request<Body>) -> Self::Future {
            let subject = request
                .extensions()
                .get::<Identity>()
                .map(|identity| identity.subject.clone())
                .unwrap_or_else(|| "no identity".to_string());
            Box::pin(async move {
                Ok(Response::builder()
                    .header("grpc-status", "0")
                    .header("x-subject", subject)
                    .body(Body::empty())
                    .unwrap())
            })
        }
    }

    /// Calls `method` through the layer and reports the gRPC status the caller
    /// would see, and the subject the service was handed.
    async fn call(
        auth: Arc<dyn Authenticator>,
        method: &str,
        authorization: Option<&str>,
    ) -> (Code, String) {
        let mut request = Request::builder().uri(method);
        if let Some(value) = authorization {
            request = request.header("authorization", value);
        }

        let response = AuthLayer::new(auth)
            .layer(Inner)
            .oneshot(request.body(Body::empty()).unwrap())
            .await
            .expect("the service answers");

        let status = Status::from_header_map(response.headers())
            .map(|status| status.code())
            .unwrap_or(Code::Unknown);
        let subject = response
            .headers()
            .get("x-subject")
            .map(|value| value.to_str().unwrap().to_string())
            .unwrap_or_default();
        (status, subject)
    }

    fn no_auth() -> Arc<dyn Authenticator> {
        Arc::new(NoAuth)
    }

    fn static_auth() -> Arc<dyn Authenticator> {
        Arc::new(StaticTokenAuth::new("s3cret"))
    }

    #[tokio::test]
    async fn the_default_lets_every_call_through_as_anonymous() {
        let (code, subject) = call(no_auth(), INFER, None).await;
        assert_eq!(code, Code::Ok);
        assert_eq!(subject, "anonymous");
    }

    #[tokio::test]
    async fn an_unchecked_token_does_not_become_an_identity() {
        let (code, subject) = call(no_auth(), INFER, Some("Bearer whatever")).await;
        assert_eq!(code, Code::Ok);
        assert_eq!(subject, "anonymous");
    }

    #[tokio::test]
    async fn a_valid_token_names_the_caller_for_the_service() {
        let (code, subject) = call(static_auth(), INFER, Some("Bearer s3cret")).await;
        assert_eq!(code, Code::Ok);
        assert_eq!(subject, "static-token");
    }

    /// The whole point of refusing with a `Status`: a client can tell a bad
    /// credential from a broken connection.
    #[tokio::test]
    async fn a_missing_token_is_refused_as_unauthenticated() {
        let (code, _) = call(static_auth(), INFER, None).await;
        assert_eq!(code, Code::Unauthenticated);
    }

    #[tokio::test]
    async fn a_wrong_or_malformed_token_is_refused_as_unauthenticated() {
        for header in ["Bearer wrong", "Basic dXNlcjpwYXNz", "Bearer", "s3cret"] {
            let (code, _) = call(static_auth(), INFER, Some(header)).await;
            assert_eq!(code, Code::Unauthenticated, "{header:?}");
        }
    }

    #[tokio::test]
    async fn a_token_that_does_not_reach_far_enough_is_permission_denied() {
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

        let (code, _) = call(Arc::new(AlwaysForbidden), INFER, Some("Bearer any")).await;
        assert_eq!(code, Code::PermissionDenied);
    }

    /// The gRPC counterpart of the HTTP carve-out: a sidecar's probe has no
    /// token either.
    #[tokio::test]
    async fn liveness_and_readiness_stay_open() {
        for method in [LIVE, MODEL_READY] {
            let (code, _) = call(static_auth(), method, None).await;
            assert_eq!(code, Code::Ok, "{method} must stay open");
        }
    }

    #[tokio::test]
    async fn metadata_and_inference_do_not() {
        for method in [
            INFER,
            "/inference.GRPCInferenceService/ModelMetadata",
            "/inference.GRPCInferenceService/ServerMetadata",
        ] {
            let (code, _) = call(static_auth(), method, None).await;
            assert_eq!(code, Code::Unauthenticated, "{method} must be closed");
        }
    }
}
