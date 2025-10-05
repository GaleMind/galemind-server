use axum::{
    Json,
    extract::FromRequestParts,
    http::{StatusCode, request::Parts},
    response::{IntoResponse, Response},
};
use serde_json::json;

#[derive(Debug, Clone, PartialEq)]
pub enum InferenceProtocol {
    OpenAI,
    Galemind,
}

impl Default for InferenceProtocol {
    fn default() -> Self {
        InferenceProtocol::Galemind
    }
}

impl<S> FromRequestParts<S> for InferenceProtocol
where
    S: Send + Sync,
{
    type Rejection = (StatusCode, Json<serde_json::Value>);

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        let protocol = parts
            .headers
            .get("X-Protocol-Inference")
            .and_then(|value| value.to_str().ok())
            .map(|s| s.to_lowercase());

        match protocol.as_deref() {
            Some("openai") => Ok(InferenceProtocol::OpenAI),
            Some("galemind") | None => Ok(InferenceProtocol::Galemind),
            Some(unknown) => Err((
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": {
                        "message": format!("Unsupported protocol: {}. Supported protocols: 'openai', 'galemind'", unknown),
                        "type": "invalid_request_error",
                        "param": "X-Protocol-Inference",
                        "code": "unsupported_protocol"
                    }
                })),
            )),
        }
    }
}

pub struct ProtocolError {
    pub message: String,
    pub status_code: StatusCode,
}

impl IntoResponse for ProtocolError {
    fn into_response(self) -> Response {
        let body = Json(json!({
            "error": {
                "message": self.message,
                "type": "protocol_error",
                "code": "protocol_mismatch"
            }
        }));
        (self.status_code, body).into_response()
    }
}
