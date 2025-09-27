use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct RestClient {
    client: Client,
    base_url: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub timestamp: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub model_name: String,
    pub input_data: Value,
    pub parameters: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub predictions: Value,
    pub model_name: String,
    pub inference_time_ms: f64,
}

impl RestClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.to_string(),
        }
    }

    pub async fn health_check(&self) -> anyhow::Result<HealthResponse> {
        let response = self
            .client
            .get(&format!("{}/health", self.base_url))
            .send()
            .await?;

        if !response.status().is_success() {
            anyhow::bail!("Health check failed with status: {}", response.status());
        }

        let health_response = response.json::<HealthResponse>().await?;
        Ok(health_response)
    }

    pub async fn list_models(&self) -> anyhow::Result<Vec<String>> {
        let response = self
            .client
            .get(&format!("{}/models", self.base_url))
            .send()
            .await?;

        if !response.status().is_success() {
            anyhow::bail!("List models failed with status: {}", response.status());
        }

        let models = response.json::<Vec<String>>().await?;
        Ok(models)
    }

    pub async fn inference(&self, request: InferenceRequest) -> anyhow::Result<InferenceResponse> {
        let response = self
            .client
            .post(&format!("{}/inference", self.base_url))
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Inference failed with status: {}, body: {}", response.status(), error_text);
        }

        let inference_response = response.json::<InferenceResponse>().await?;
        Ok(inference_response)
    }

    pub async fn model_info(&self, model_name: &str) -> anyhow::Result<Value> {
        let response = self
            .client
            .get(&format!("{}/models/{}", self.base_url, model_name))
            .send()
            .await?;

        if !response.status().is_success() {
            anyhow::bail!("Model info failed with status: {}", response.status());
        }

        let model_info = response.json::<Value>().await?;
        Ok(model_info)
    }
}