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

    // Not "unsupported model artifact": the variant also covers a call a
    // transport does not implement, and a message reading "unsupported model
    // artifact: the grpc transport has no model catalogue call" names the
    // wrong thing. The detail says what is unsupported.
    #[error("unsupported: {0}")]
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

    /// A failure a remote server reported, kept exactly as it described it.
    ///
    /// A client must not re-word what the server said: the server knows what
    /// went wrong, and wrapping its sentence inside another one only produces
    /// nested quoting. The flag records whether the request or the server was
    /// at fault, which is the part a caller may still want to branch on.
    #[error("{message}")]
    Remote { message: String, client_error: bool },
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
                | Self::Remote {
                    client_error: true,
                    ..
                }
        )
    }

    /// A failure reported by a remote server, blamed on the request.
    pub fn remote_client_error(message: impl Into<String>) -> Self {
        Self::Remote {
            message: message.into(),
            client_error: true,
        }
    }

    /// A failure reported by a remote server, blamed on the server.
    pub fn remote_server_error(message: impl Into<String>) -> Self {
        Self::Remote {
            message: message.into(),
            client_error: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_remote_failure_is_shown_exactly_as_the_server_worded_it() {
        let error = InferError::remote_client_error("model 'ghost' is not known to this server");
        assert_eq!(
            error.to_string(),
            "model 'ghost' is not known to this server"
        );
        assert!(error.is_client_error());
        assert!(!error.is_retryable());
    }

    #[test]
    fn a_remote_server_failure_is_not_the_requests_fault() {
        assert!(!InferError::remote_server_error("boom").is_client_error());
    }

    #[test]
    fn overload_and_timeout_are_the_retryable_failures() {
        assert!(InferError::Overloaded.is_retryable());
        assert!(InferError::Timeout(std::time::Duration::from_secs(1)).is_retryable());
        assert!(!InferError::Backend("boom".into()).is_retryable());
    }
}
