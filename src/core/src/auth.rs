//! Authentication, as a seam rather than a decision.
//!
//! The server needs somewhere for an identity check to go before one is
//! chosen, and the choice is a deployment's: an internal server behind a mesh
//! wants none, a shared one wants tokens from the organisation's identity
//! provider. So authentication is a trait, and the default implementation,
//! [`NoAuth`], lets everything through.
//!
//! The trait lives here, in the domain, rather than in a transport, because
//! both transports have to make the same decision about the same token and an
//! identity provider is not an HTTP detail. What a transport adds is the part
//! that is its own: parsing a header, shaping a refusal, and deciding which
//! of its own endpoints are exempt. Those exemptions are listed here too --
//! [`is_open_http_path`] and [`is_open_grpc_method`] -- so that the two ports
//! cannot drift into protecting different things.
//!
//! # Adding a provider
//!
//! Implement [`Authenticator`] and pass it to `rest_server::app_with_auth` and
//! `grpc_server::serve_with_auth`. The trait takes the bearer token and
//! returns an [`Identity`] or an [`AuthError`]; nothing else in the server
//! changes, and the wiring is one line in the binary.
//!
//! A Keycloak implementation is the case this was shaped for. It would hold
//! the realm's issuer URL, the audience it accepts and a cache of the realm's
//! JWKS, and `authenticate` would verify the token's signature against the
//! cached key for its `kid`, check `iss`, `aud` and `exp`, and return the
//! `sub` with the realm roles as scopes -- refusing with
//! [`AuthError::Invalid`] on a bad signature or an expired token, and
//! refetching the JWKS when a `kid` is unknown. Because `authenticate` is
//! async, fetching a key it has not cached is an await rather than a blocking
//! call on a request thread. It needs a JWT library and a key cache, which is
//! why it is not here yet: this is the seam it will plug into, not a
//! placeholder for it.

use async_trait::async_trait;

/// Who a request is from.
///
/// Put in the request's extensions by whichever transport checked it, so a
/// handler can read it. Nothing in the protocol's handlers reads it yet; it is
/// here so that per-model authorization or per-tenant metrics can be added
/// without touching the middleware again.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Identity {
    /// Who the caller is, as the provider names them. `anonymous` when no
    /// check was made.
    pub subject: String,
    /// What they are allowed to do, as the provider reports it: OAuth scopes,
    /// Keycloak realm roles, whatever the provider's vocabulary is.
    pub scopes: Vec<String>,
    /// Who vouched for them, when the provider says.
    pub issuer: Option<String>,
}

impl Identity {
    /// The identity of a request nobody checked.
    pub fn anonymous() -> Self {
        Self {
            subject: "anonymous".to_string(),
            scopes: Vec::new(),
            issuer: None,
        }
    }

    /// A named caller with no scopes, which is what a shared-secret check can
    /// honestly report.
    pub fn subject(subject: impl Into<String>) -> Self {
        Self {
            subject: subject.into(),
            scopes: Vec::new(),
            issuer: None,
        }
    }

    pub fn is_anonymous(&self) -> bool {
        self.subject == "anonymous"
    }

    pub fn has_scope(&self, scope: &str) -> bool {
        self.scopes.iter().any(|held| held == scope)
    }
}

/// Why a request was refused.
///
/// Split by what the caller should do about it: the first three mean "your
/// credential is the problem", which is a 401 and gRPC's `UNAUTHENTICATED`,
/// and the last means "your credential is fine and does not permit this",
/// which is a 403 and gRPC's `PERMISSION_DENIED`. Conflating them sends a
/// caller to re-authenticate over and over against a permission they will
/// never be granted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthError {
    /// No `Authorization` header at all.
    Missing,
    /// An `Authorization` header that is not a bearer token.
    Malformed(String),
    /// A bearer token that did not check out.
    Invalid(String),
    /// A valid token that does not permit this request.
    Forbidden(String),
}

impl AuthError {
    /// Whether the credential was accepted but does not reach far enough.
    ///
    /// The one distinction every transport has to make, and the only reason
    /// to match on this type: each maps it to its own vocabulary.
    pub fn is_forbidden(&self) -> bool {
        matches!(self, Self::Forbidden(_))
    }

    /// The HTTP status this refusal deserves.
    pub fn http_status(&self) -> u16 {
        if self.is_forbidden() {
            403
        } else {
            401
        }
    }

    /// What the caller is told.
    ///
    /// Deliberately short of detail: why a token failed to verify is useful to
    /// an operator in the log and useful to an attacker in a response body.
    pub fn message(&self) -> &'static str {
        match self {
            Self::Missing => "this server requires an Authorization: Bearer token",
            Self::Malformed(_) => "the Authorization header must be 'Bearer <token>'",
            Self::Invalid(_) => "the bearer token was not accepted",
            Self::Forbidden(_) => "the bearer token does not permit this request",
        }
    }

    /// The detail for the server's own log.
    pub fn detail(&self) -> &str {
        match self {
            Self::Missing => "no Authorization header",
            Self::Malformed(detail) | Self::Invalid(detail) | Self::Forbidden(detail) => detail,
        }
    }
}

impl std::fmt::Display for AuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.message(), self.detail())
    }
}

impl std::error::Error for AuthError {}

/// Checks a bearer token and says who sent it.
///
/// `token` is `None` when the request carried no `Authorization` header at
/// all, which an implementation may accept -- that is how a mixed deployment
/// serves anonymous reads and authenticated inference.
#[async_trait]
pub trait Authenticator: Send + Sync {
    /// How this check appears in the startup log and the metrics.
    fn name(&self) -> &'static str;

    /// Whether a request with no credential at all can be served, letting a
    /// transport refuse it without a call.
    fn allows_anonymous(&self) -> bool {
        false
    }

    async fn authenticate(&self, token: Option<&str>) -> Result<Identity, AuthError>;
}

/// Accepts every request without looking at it.
///
/// The default, and what the server runs with unless a deployment configures
/// otherwise. A token that is sent is ignored rather than trusted: reporting
/// an unverified token's subject as an identity would make an unauthenticated
/// server look like an authenticated one in the log.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoAuth;

#[async_trait]
impl Authenticator for NoAuth {
    fn name(&self) -> &'static str {
        "none"
    }

    fn allows_anonymous(&self) -> bool {
        true
    }

    async fn authenticate(&self, _token: Option<&str>) -> Result<Identity, AuthError> {
        Ok(Identity::anonymous())
    }
}

/// Accepts one shared secret, compared in constant time.
///
/// Enough for a server reachable only by a known client, and a second
/// implementation of the trait, which is what keeps the seam honest: a
/// transport cannot quietly depend on authentication always succeeding. It is
/// not an identity provider -- every caller is the same subject, and there is
/// nothing to revoke but the secret itself.
pub struct StaticTokenAuth {
    token: String,
    subject: String,
}

impl StaticTokenAuth {
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            token: token.into(),
            subject: "static-token".to_string(),
        }
    }

    /// Names the caller this token stands for, for the log.
    pub fn with_subject(mut self, subject: impl Into<String>) -> Self {
        self.subject = subject.into();
        self
    }
}

#[async_trait]
impl Authenticator for StaticTokenAuth {
    fn name(&self) -> &'static str {
        "static-token"
    }

    async fn authenticate(&self, token: Option<&str>) -> Result<Identity, AuthError> {
        let token = token.ok_or(AuthError::Missing)?;
        if constant_time_eq(token.as_bytes(), self.token.as_bytes()) {
            Ok(Identity::subject(&self.subject))
        } else {
            Err(AuthError::Invalid("the token does not match".to_string()))
        }
    }
}

/// Compares two secrets without leaking where they first differ.
///
/// `==` on a slice returns as soon as a byte differs, which over many attempts
/// reveals a prefix. The length is not secret -- it leaks through the response
/// either way -- so differing lengths may return early.
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b)
        .fold(0u8, |differences, (x, y)| differences | (x ^ y))
        == 0
}

/// Reads the bearer token from an `Authorization` header's value.
///
/// The scheme is matched case-insensitively, as RFC 9110 requires.
pub fn bearer_token(header: &str) -> Result<&str, AuthError> {
    let (scheme, token) = header.split_once(' ').ok_or_else(|| {
        AuthError::Malformed("the Authorization header has no scheme".to_string())
    })?;
    if !scheme.eq_ignore_ascii_case("bearer") {
        return Err(AuthError::Malformed(format!(
            "the {scheme} scheme is not supported"
        )));
    }
    let token = token.trim();
    if token.is_empty() {
        return Err(AuthError::Malformed(
            "the bearer token is empty".to_string(),
        ));
    }
    Ok(token)
}

/// HTTP paths served without authentication, whatever is configured.
///
/// Health and metrics, and nothing else. A liveness or readiness probe is run
/// by an orchestrator that has no token to send, and a metrics scrape by
/// Prometheus, which has no way to send one; requiring a token on these turns
/// switching authentication on into a rolling restart of a server that then
/// reports itself unhealthy.
///
/// Every readiness endpoint counts, including a model's own: a probe that
/// waits for one named model to be ready is how a deployment that preloads
/// nothing avoids sending traffic into a cold cache, and that probe has no
/// more credentials than the server-wide one. What it can tell an
/// unauthenticated caller is whether a model it already knew the name of is
/// loaded -- a boolean, from a name it had to supply -- and not what models
/// exist, what they take or what they answer. Those are the catalogue and
/// metadata endpoints, which stay closed.
///
/// A deployment that considers even that too much keeps the port off the
/// public network; one that considers the counters sensitive turns the scrape
/// off with `--allow-metrics false`.
pub fn is_open_http_path(path: &str) -> bool {
    matches!(path, "/metrics" | "/v2/health/live" | "/v2/health/ready") || is_model_ready_path(path)
}

/// Whether a path is a model readiness check.
///
/// Matched on shape rather than by prefix, so that only the two readiness
/// URLs the protocol defines are open:
/// `/v2/models/{model}/ready` and
/// `/v2/models/{model}/versions/{version}/ready`. A model literally named
/// `ready` cannot widen this, because the segment count still has to match.
fn is_model_ready_path(path: &str) -> bool {
    let Some(rest) = path.strip_prefix("/v2/models/") else {
        return false;
    };
    let segments: Vec<&str> = rest.split('/').collect();
    match segments.as_slice() {
        [model, "ready"] => !model.is_empty(),
        [model, "versions", version, "ready"] => !model.is_empty() && !version.is_empty(),
        _ => false,
    }
}

/// gRPC methods served without authentication, whatever is configured.
///
/// The gRPC counterparts of [`is_open_http_path`]: liveness, readiness and
/// model readiness. `method` is the full path tonic routes on, for example
/// `/inference.GRPCInferenceService/ServerLive`. Metadata and inference are
/// not here, so a closed server stays closed on both ports.
pub fn is_open_grpc_method(method: &str) -> bool {
    matches!(
        method,
        "/inference.GRPCInferenceService/ServerLive"
            | "/inference.GRPCInferenceService/ServerReady"
            | "/inference.GRPCInferenceService/ModelReady"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secrets_are_compared_without_revealing_where_they_differ() {
        assert!(constant_time_eq(b"s3cret", b"s3cret"));
        assert!(!constant_time_eq(b"s3cret", b"s3cres"));
        assert!(!constant_time_eq(b"s3cret", b"s3cret-longer"));
        assert!(constant_time_eq(b"", b""));
    }

    #[test]
    fn a_bearer_header_yields_its_token_whatever_the_schemes_case() {
        assert_eq!(bearer_token("Bearer s3cret"), Ok("s3cret"));
        assert_eq!(bearer_token("bearer s3cret"), Ok("s3cret"));
        assert_eq!(bearer_token("BEARER  s3cret  "), Ok("s3cret"));
    }

    #[test]
    fn anything_that_is_not_a_bearer_token_is_malformed() {
        for header in ["Basic dXNlcjpwYXNz", "Bearer", "Bearer   ", "s3cret", ""] {
            let error = bearer_token(header).expect_err(header);
            assert!(matches!(error, AuthError::Malformed(_)), "{header:?}");
            assert_eq!(error.http_status(), 401);
        }
    }

    #[test]
    fn a_forbidden_credential_is_not_a_request_to_authenticate_again() {
        let forbidden = AuthError::Forbidden("no roles".to_string());
        assert!(forbidden.is_forbidden());
        assert_eq!(forbidden.http_status(), 403);

        for error in [
            AuthError::Missing,
            AuthError::Malformed("x".to_string()),
            AuthError::Invalid("x".to_string()),
        ] {
            assert!(!error.is_forbidden());
            assert_eq!(error.http_status(), 401);
        }
    }

    /// The reason is for the log; the caller gets the generic message.
    #[test]
    fn a_refusal_keeps_its_reason_out_of_what_the_caller_is_told() {
        let error = AuthError::Invalid("signature does not verify for kid=abc".to_string());
        assert_eq!(error.message(), "the bearer token was not accepted");
        assert!(error.detail().contains("kid=abc"));
    }

    #[tokio::test]
    async fn the_default_accepts_everything_as_anonymous() {
        let auth = NoAuth;
        assert!(auth.allows_anonymous());
        assert!(auth.authenticate(None).await.unwrap().is_anonymous());
        // A token nobody checked must not become an identity.
        assert!(auth
            .authenticate(Some("whatever"))
            .await
            .unwrap()
            .is_anonymous());
    }

    #[tokio::test]
    async fn a_shared_secret_names_its_caller_and_refuses_everything_else() {
        let auth = StaticTokenAuth::new("s3cret").with_subject("batch-job");
        assert!(!auth.allows_anonymous());

        let identity = auth.authenticate(Some("s3cret")).await.unwrap();
        assert_eq!(identity.subject, "batch-job");
        assert!(!identity.is_anonymous());

        assert_eq!(auth.authenticate(None).await, Err(AuthError::Missing));
        assert!(matches!(
            auth.authenticate(Some("wrong")).await,
            Err(AuthError::Invalid(_))
        ));
    }

    /// Both transports have to exempt health and metrics, and nothing else:
    /// a probe has no token, and a catalogue listing is not a probe.
    #[test]
    fn health_and_metrics_are_open_on_both_transports() {
        for path in [
            "/metrics",
            "/v2/health/live",
            "/v2/health/ready",
            "/v2/models/thermal_cnn/ready",
            "/v2/models/thermal_cnn/versions/2/ready",
        ] {
            assert!(is_open_http_path(path), "{path} must stay open");
        }

        for method in [
            "/inference.GRPCInferenceService/ServerLive",
            "/inference.GRPCInferenceService/ServerReady",
            "/inference.GRPCInferenceService/ModelReady",
        ] {
            assert!(is_open_grpc_method(method), "{method} must stay open");
        }
    }

    #[test]
    fn everything_that_says_what_a_model_is_stays_closed() {
        for path in [
            "/v2",
            "/v2/models/list",
            "/v2/models/info/thermal_cnn",
            "/v2/models/thermal_cnn",
            "/v2/models/thermal_cnn/infer",
            "/v2/models/thermal_cnn/versions/2",
            "/v2/models/thermal_cnn/versions/2/infer",
            // A readiness URL's shape, worn by something that is not one.
            "/v2/models//ready",
            "/v2/models/thermal_cnn/ready/extra",
            "/v2/models/thermal_cnn/notversions/2/ready",
            "/metrics/../v2/models/list",
        ] {
            assert!(!is_open_http_path(path), "{path} must be authenticated");
        }

        for method in [
            "/inference.GRPCInferenceService/ModelInfer",
            "/inference.GRPCInferenceService/ModelMetadata",
            "/inference.GRPCInferenceService/ServerMetadata",
            "/grpc.health.v1.Health/Check",
        ] {
            assert!(
                !is_open_grpc_method(method),
                "{method} must be authenticated"
            );
        }
    }

    #[test]
    fn scopes_are_what_an_implementation_reports_them_to_be() {
        let identity = Identity {
            subject: "svc-batch".to_string(),
            scopes: vec!["infer".to_string()],
            issuer: Some("https://keycloak.example/realms/ml".to_string()),
        };
        assert!(identity.has_scope("infer"));
        assert!(!identity.has_scope("admin"));
        assert!(!identity.is_anonymous());
    }
}
