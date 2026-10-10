//! The single error type crossing every layer of the server.

use thiserror::Error;

/// Failures that can occur while serving an inference, metadata or health request.
///
/// Transport crates map these onto HTTP status codes and gRPC status codes; the
/// domain itself stays free of any transport dependency. The type is `Clone`
/// because one failed batch has to be reported to every request in it.
#[derive(Debug, Clone, Error)]
pub enum InferError {
    #[error("model '{0}' is not known to this server")]
    ModelNotFound(String),

    #[error("model '{name}' has no version '{version}'")]
    VersionNotFound { name: String, version: String },

    #[error("model '{0}' is not ready for inference")]
    ModelNotReady(String),

    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("unsupported model artifact: {0}")]
    Unsupported(String),

    #[error("backend failure: {0}")]
    Backend(String),

    #[error("model store failure: {0}")]
    Store(String),

    #[error("server is overloaded, retry later")]
    Overloaded,

    #[error("request timed out after {0:?}")]
    Timeout(std::time::Duration),

    #[error("internal error: {0}")]
    Internal(String),
}

impl InferError {
    /// `true` when the client can reasonably retry the very same request.
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::Overloaded | Self::Timeout(_))
    }

    /// `true` when the request itself was at fault rather than the server.
    pub fn is_client_error(&self) -> bool {
        matches!(
            self,
            Self::ModelNotFound(_)
                | Self::VersionNotFound { .. }
                | Self::InvalidRequest(_)
                | Self::Unsupported(_)
        )
    }
}
