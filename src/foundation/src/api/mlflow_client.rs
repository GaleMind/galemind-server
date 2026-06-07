use anyhow::{anyhow, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLFlowModel {
    pub name: String,
    pub version: Option<String>,
    pub creation_timestamp: Option<i64>,
    pub last_updated_timestamp: Option<i64>,
    pub description: Option<String>,
    pub tags: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLFlowModelVersion {
    pub name: String,
    pub version: String,
    pub creation_timestamp: Option<i64>,
    pub last_updated_timestamp: Option<i64>,
    pub description: Option<String>,
    pub user_id: Option<String>,
    pub current_stage: Option<String>,
    pub source: Option<String>,
    pub run_id: Option<String>,
    pub status: Option<String>,
    pub tags: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ListModelsResponse {
    registered_models: Vec<MLFlowModel>,
    next_page_token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GetModelVersionsResponse {
    model_versions: Vec<MLFlowModelVersion>,
    next_page_token: Option<String>,
}

#[async_trait]
pub trait MLFlowClientTrait: Send + Sync {
    async fn list_models(&self) -> Result<Vec<MLFlowModel>>;
    async fn get_model_versions(&self, model_name: &str) -> Result<Vec<MLFlowModelVersion>>;
    async fn get_model(&self, name: &str) -> Result<Option<MLFlowModel>>;
}

#[derive(Debug, Clone)]
pub struct MLFlowClient {
    base_url: String,
    client: Client,
    api_token: Option<String>,
}

impl MLFlowClient {
    pub fn new(base_url: String, api_token: Option<String>) -> Self {
        Self {
            base_url,
            client: Client::new(),
            api_token,
        }
    }

    fn build_request(&self, endpoint: &str) -> reqwest::RequestBuilder {
        let url = format!(
            "{}/api/2.0/mlflow/{}",
            self.base_url.trim_end_matches('/'),
            endpoint
        );
        let mut request = self.client.get(&url);

        if let Some(token) = &self.api_token {
            request = request.header("Authorization", format!("Bearer {}", token));
        }

        request
    }

    async fn get_all_pages<T, F>(&self, endpoint: &str, extract_items: F) -> Result<Vec<T>>
    where
        F: Fn(&str) -> Result<(Vec<T>, Option<String>)>,
    {
        let mut all_items = Vec::new();
        let mut next_page_token: Option<String> = None;

        loop {
            let mut url = endpoint.to_string();
            if let Some(token) = &next_page_token {
                url.push_str(&format!("&page_token={}", token));
            }

            let response = self.build_request(&url).send().await?;

            if !response.status().is_success() {
                return Err(anyhow!(
                    "MLFlow API request failed with status: {}, body: {}",
                    response.status(),
                    response.text().await.unwrap_or_default()
                ));
            }

            let text = response.text().await?;
            let (items, next_token) = extract_items(&text)?;

            all_items.extend(items);

            next_page_token = next_token;
            if next_page_token.is_none() {
                break;
            }
        }

        Ok(all_items)
    }
}

#[async_trait]
impl MLFlowClientTrait for MLFlowClient {
    async fn list_models(&self) -> Result<Vec<MLFlowModel>> {
        self.get_all_pages("registered-models/list?max_results=100", |text| {
            let response: ListModelsResponse = serde_json::from_str(text)?;
            Ok((response.registered_models, response.next_page_token))
        })
        .await
    }

    async fn get_model_versions(&self, model_name: &str) -> Result<Vec<MLFlowModelVersion>> {
        self.get_all_pages(
            &format!(
                "model-versions/search?filter=name%3D%27{}%27&max_results=100",
                urlencoding::encode(model_name)
            ),
            |text| {
                let response: GetModelVersionsResponse = serde_json::from_str(text)?;
                Ok((response.model_versions, response.next_page_token))
            },
        )
        .await
    }

    async fn get_model(&self, name: &str) -> Result<Option<MLFlowModel>> {
        let endpoint = format!("registered-models/get?name={}", urlencoding::encode(name));
        let response = self.build_request(&endpoint).send().await?;

        if response.status().is_success() {
            #[derive(Deserialize)]
            struct GetModelResponse {
                registered_model: MLFlowModel,
            }

            let response_data: GetModelResponse = response.json().await?;
            Ok(Some(response_data.registered_model))
        } else if response.status() == reqwest::StatusCode::NOT_FOUND {
            Ok(None)
        } else {
            Err(anyhow!(
                "MLFlow API request failed with status: {}, body: {}",
                response.status(),
                response.text().await.unwrap_or_default()
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockMLFlowClient {
        models: Vec<MLFlowModel>,
        model_versions: HashMap<String, Vec<MLFlowModelVersion>>,
    }

    impl MockMLFlowClient {
        fn new() -> Self {
            let mut model_versions = HashMap::new();
            model_versions.insert(
                "test_model".to_string(),
                vec![MLFlowModelVersion {
                    name: "test_model".to_string(),
                    version: "1".to_string(),
                    creation_timestamp: Some(1234567890),
                    last_updated_timestamp: Some(1234567890),
                    description: Some("Test version".to_string()),
                    user_id: Some("test_user".to_string()),
                    current_stage: Some("Production".to_string()),
                    source: Some("/path/to/model".to_string()),
                    run_id: Some("run123".to_string()),
                    status: Some("READY".to_string()),
                    tags: Some(HashMap::new()),
                }],
            );

            Self {
                models: vec![MLFlowModel {
                    name: "test_model".to_string(),
                    version: Some("1".to_string()),
                    creation_timestamp: Some(1234567890),
                    last_updated_timestamp: Some(1234567890),
                    description: Some("Test model".to_string()),
                    tags: Some(HashMap::new()),
                }],
                model_versions,
            }
        }
    }

    #[async_trait]
    impl MLFlowClientTrait for MockMLFlowClient {
        async fn list_models(&self) -> Result<Vec<MLFlowModel>> {
            Ok(self.models.clone())
        }

        async fn get_model_versions(&self, model_name: &str) -> Result<Vec<MLFlowModelVersion>> {
            Ok(self
                .model_versions
                .get(model_name)
                .cloned()
                .unwrap_or_default())
        }

        async fn get_model(&self, name: &str) -> Result<Option<MLFlowModel>> {
            Ok(self.models.iter().find(|m| m.name == name).cloned())
        }
    }

    #[tokio::test]
    async fn test_mlflow_client_list_models() {
        let client = MockMLFlowClient::new();
        let models = client.list_models().await.unwrap();

        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "test_model");
        assert_eq!(models[0].version, Some("1".to_string()));
    }

    #[tokio::test]
    async fn test_mlflow_client_get_model() {
        let client = MockMLFlowClient::new();
        let model = client.get_model("test_model").await.unwrap();

        assert!(model.is_some());
        let model = model.unwrap();
        assert_eq!(model.name, "test_model");
    }

    #[tokio::test]
    async fn test_mlflow_client_get_nonexistent_model() {
        let client = MockMLFlowClient::new();
        let model = client.get_model("nonexistent").await.unwrap();

        assert!(model.is_none());
    }

    #[tokio::test]
    async fn test_mlflow_client_get_model_versions() {
        let client = MockMLFlowClient::new();
        let versions = client.get_model_versions("test_model").await.unwrap();

        assert_eq!(versions.len(), 1);
        assert_eq!(versions[0].name, "test_model");
        assert_eq!(versions[0].version, "1");
        assert_eq!(versions[0].current_stage, Some("Production".to_string()));
    }

    #[tokio::test]
    async fn test_mlflow_client_get_versions_for_nonexistent_model() {
        let client = MockMLFlowClient::new();
        let versions = client.get_model_versions("nonexistent").await.unwrap();

        assert!(versions.is_empty());
    }

    #[test]
    fn test_mlflow_client_creation() {
        let client = MLFlowClient::new(
            "http://localhost:5000".to_string(),
            Some("token123".to_string()),
        );
        assert_eq!(client.base_url, "http://localhost:5000");
        assert_eq!(client.api_token, Some("token123".to_string()));
    }

    #[test]
    fn test_mlflow_client_creation_without_token() {
        let client = MLFlowClient::new("http://localhost:5000".to_string(), None);
        assert_eq!(client.base_url, "http://localhost:5000");
        assert_eq!(client.api_token, None);
    }
}
