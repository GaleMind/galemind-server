//! Turning domain errors into the protocol's HTTP responses.

use crate::wire::ErrorBody;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use galemind_core::InferError;

/// A domain error rendered as the protocol's error object.
///
/// Every failing endpoint answers with `{"error": "..."}` and an HTTP error
/// status, which is what the specification requires of the metadata and
/// inference endpoints alike.
pub struct ApiError(pub InferError);

impl From<InferError> for ApiError {
    fn from(error: InferError) -> Self {
        Self(error)
    }
}

/// The status code for a failure.
///
/// The specification only says a failure must be an HTTP error status, giving
/// 400 as typical, so these follow ordinary HTTP meanings to stay useful to
/// clients and proxies.
pub fn status_for(error: &InferError) -> StatusCode {
    match error {
        InferError::ModelNotFound(_) | InferError::VersionNotFound { .. } => StatusCode::NOT_FOUND,
        InferError::InvalidRequest(_) => StatusCode::BAD_REQUEST,
        InferError::Unsupported(_) => StatusCode::NOT_IMPLEMENTED,
        InferError::ModelNotReady(_) | InferError::Overloaded => StatusCode::SERVICE_UNAVAILABLE,
        InferError::Timeout(_) => StatusCode::GATEWAY_TIMEOUT,
        InferError::Backend(_) | InferError::Store(_) | InferError::Internal(_) => {
            StatusCode::INTERNAL_SERVER_ERROR
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let status = status_for(&self.0);
        if status.is_server_error() {
            tracing::error!(error = %self.0, "request failed");
        } else {
            tracing::debug!(error = %self.0, "request refused");
        }
        (
            status,
            axum::Json(ErrorBody {
                error: self.0.to_string(),
            }),
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn failures_map_onto_the_expected_http_statuses() {
        let cases = [
            (InferError::ModelNotFound("m".into()), StatusCode::NOT_FOUND),
            (
                InferError::VersionNotFound {
                    name: "m".into(),
                    version: "2".into(),
                },
                StatusCode::NOT_FOUND,
            ),
            (
                InferError::InvalidRequest("bad".into()),
                StatusCode::BAD_REQUEST,
            ),
            (InferError::Overloaded, StatusCode::SERVICE_UNAVAILABLE),
            (
                InferError::Timeout(Duration::from_secs(1)),
                StatusCode::GATEWAY_TIMEOUT,
            ),
            (
                InferError::Backend("boom".into()),
                StatusCode::INTERNAL_SERVER_ERROR,
            ),
            (
                InferError::Unsupported("nope".into()),
                StatusCode::NOT_IMPLEMENTED,
            ),
        ];
        for (error, expected) in cases {
            assert_eq!(status_for(&error), expected, "{error}");
        }
    }
}
