use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use foundation::ModelDiscoveryService;
use serde_json::json;

use crate::{
    data_model::{InferenceRequest, InferenceResponse, MetadataModelResponse, MetadataTensor},
    openai_models::{
        ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ModelInfo,
        ModelsListResponse, Usage,
    },
    protocol::InferenceProtocol,
};

async fn unified_chat_completions(
    protocol: InferenceProtocol,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    match protocol {
        InferenceProtocol::OpenAI => handle_openai_chat_completions(payload).await,
        InferenceProtocol::Galemind => handle_galemind_inference(payload).await,
    }
}

async fn handle_openai_chat_completions(
    payload: serde_json::Value,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let request: ChatCompletionRequest = serde_json::from_value(payload).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": {
                    "message": format!("Invalid request format: {}", e),
                    "type": "invalid_request_error",
                    "param": null,
                    "code": "invalid_json"
                }
            })),
        )
    })?;

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4().simple()),
        object: "chat.completion".to_string(),
        created: now,
        model: request.model.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content:
                    "Hello! This is a test response from the Galemind server using OpenAI protocol."
                        .to_string(),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            finish_reason: "stop".to_string(),
            logprobs: None,
        }],
        usage: Usage {
            prompt_tokens: 10,
            completion_tokens: 15,
            total_tokens: 25,
        },
        system_fingerprint: Some("fp_galemind_v1".to_string()),
    };

    Ok(Json(serde_json::to_value(response).unwrap()))
}

async fn handle_galemind_inference(
    payload: serde_json::Value,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let request: InferenceRequest = serde_json::from_value(payload).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": format!("Invalid request format: {}", e)
            })),
        )
    })?;

    let response = InferenceResponse {
        id: request.id,
        outputs: Some(vec![MetadataTensor {
            name: "response_tensor".to_string(),
            shape: vec![1, 100],
            datatype: "string".to_string(),
            parameters: None,
            data: None,
        }]),
    };

    Ok(Json(serde_json::to_value(response).unwrap()))
}

async fn unified_models_list(
    protocol: InferenceProtocol,
    State(model_manager): State<Arc<ModelDiscoveryService>>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    match protocol {
        InferenceProtocol::OpenAI => handle_openai_models_list(model_manager).await,
        InferenceProtocol::Galemind => handle_galemind_models_list(model_manager).await,
    }
}

async fn handle_openai_models_list(
    _model_manager: Arc<ModelDiscoveryService>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let response = ModelsListResponse {
        object: "list".to_string(),
        data: vec![
            ModelInfo {
                id: "gpt-3.5-turbo".to_string(),
                object: "model".to_string(),
                created: now,
                owned_by: "galemind".to_string(),
                permission: None,
                root: None,
                parent: None,
            },
            ModelInfo {
                id: "gpt-4".to_string(),
                object: "model".to_string(),
                created: now,
                owned_by: "galemind".to_string(),
                permission: None,
                root: None,
                parent: None,
            },
        ],
    };

    Ok(Json(serde_json::to_value(response).unwrap()))
}

async fn handle_galemind_models_list(
    _model_manager: Arc<ModelDiscoveryService>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let tensor = MetadataTensor {
        name: "model_tensor".to_string(),
        shape: vec![12, 21],
        datatype: "magic".to_string(),
        parameters: None,
        data: None,
    };

    let response = MetadataModelResponse {
        name: "galemind-model".to_string(),
        versions: Some(vec!["v1".to_string(), "v2".to_string()]),
        platform: vec!["galemind_platform".to_string()],
        inputs: vec![tensor.clone()],
        outputs: vec![tensor],
    };

    Ok(Json(serde_json::to_value(response).unwrap()))
}

async fn unified_model_ready(
    protocol: InferenceProtocol,
    Path(model_name): Path<String>,
) -> impl IntoResponse {
    match protocol {
        InferenceProtocol::OpenAI => Json(json!({
            "status": "ready",
            "model": model_name,
            "protocol": "openai"
        })),
        InferenceProtocol::Galemind => Json(json!({
            "ready": true,
            "model": model_name,
            "protocol": "galemind"
        })),
    }
}

pub fn new_unified_router(model_manager: Arc<ModelDiscoveryService>) -> Router {
    Router::new()
        .route("/chat/completions", post(unified_chat_completions))
        .route("/models", get(unified_models_list))
        .route("/models/{model_name}/ready", get(unified_model_ready))
        .with_state(model_manager)
}
