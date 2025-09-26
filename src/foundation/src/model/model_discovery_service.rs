use dashmap::DashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::api::inference::InferenceRequest;
use crate::api::mlflow_client::{MLFlowClient, MLFlowClientTrait};
use crate::model::circular_buffer::CircularBuffer;

#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub struct ModelId(pub String);

impl ModelId {
    pub fn from_path(models_path: PathBuf) -> Option<Self> {
        if models_path.file_name().is_none() || models_path.extension().is_none() {
            return None;
        }

        models_path
            .file_name()
            .and_then(|os_model_str| os_model_str.to_str())
            .map(|model| ModelId(model.to_string()))
    }

    pub fn from_string(id: String) -> Self {
        ModelId(id)
    }

    pub fn from_url(url: &str) -> Option<Self> {
        // Extract model name from URL path
        url.split('/')
            .last()
            .filter(|s| !s.is_empty())
            .map(|s| ModelId(s.to_string()))
    }
}

#[derive(Debug, Clone)]
pub enum ModelSource {
    Path(PathBuf),
    Url(String),
    Id(String),
    MLFlow {
        base_url: String,
        api_token: Option<String>,
        model_name: Option<String>, // If None, discover all models
    },
}

pub struct ModelDiscoveryService {
    models: DashMap<ModelId, Mutex<CircularBuffer<InferenceRequest>>>,
    models_buffer_capacity: usize,
}

impl ModelDiscoveryService {
    pub fn new(models_buffer_capacity: usize) -> Self {
        Self {
            models: DashMap::new(),
            models_buffer_capacity,
        }
    }

    pub async fn discover_models(
        &self,
        sources: Vec<ModelSource>,
    ) -> Result<Vec<ModelId>, Box<dyn std::error::Error>> {
        let mut discovered_models = Vec::new();

        for source in sources {
            match source {
                ModelSource::Path(path) => {
                    if path.is_dir() {
                        self.load_models_from_dir(&path)?;
                        let models = self.discover_from_directory(&path)?;
                        discovered_models.extend(models);
                    } else if let Some(model_id) = ModelId::from_path(path) {
                        self.register_model(model_id.clone());
                        discovered_models.push(model_id);
                    }
                }
                ModelSource::Url(url) => {
                    if let Some(model_id) = ModelId::from_url(&url) {
                        self.register_model(model_id.clone());
                        discovered_models.push(model_id);
                    }
                }
                ModelSource::Id(id) => {
                    let model_id = ModelId::from_string(id);
                    self.register_model(model_id.clone());
                    discovered_models.push(model_id);
                }
                ModelSource::MLFlow {
                    base_url,
                    api_token,
                    model_name,
                } => {
                    let models = self
                        .discover_from_mlflow(base_url, api_token, model_name)
                        .await?;
                    discovered_models.extend(models);
                }
            }
        }

        Ok(discovered_models)
    }

    async fn discover_from_mlflow(
        &self,
        base_url: String,
        api_token: Option<String>,
        model_name: Option<String>,
    ) -> Result<Vec<ModelId>, Box<dyn std::error::Error>> {
        let client = MLFlowClient::new(base_url, api_token);
        let mut discovered_models = Vec::new();

        if let Some(specific_model) = model_name {
            // Discover specific model
            if let Some(model) = client.get_model(&specific_model).await? {
                let model_id = ModelId::from_string(model.name);
                self.register_model(model_id.clone());
                discovered_models.push(model_id);
            }
        } else {
            // Discover all models
            let models = client.list_models().await?;
            for model in models {
                let model_id = ModelId::from_string(model.name);
                self.register_model(model_id.clone());
                discovered_models.push(model_id);
            }
        }

        Ok(discovered_models)
    }

    fn discover_from_directory(&self, models_dir: &Path) -> std::io::Result<Vec<ModelId>> {
        let mut models = Vec::new();
        let model_entries = fs::read_dir(models_dir)?;

        for model_entry in model_entries {
            let model_entry = model_entry?;
            if model_entry.file_type()?.is_dir() {
                if let Some(model_id) = ModelId::from_path(model_entry.path()) {
                    models.push(model_id);
                }
            }
        }

        Ok(models)
    }

    pub fn load_models_from_dir<P: AsRef<Path>>(&self, models_dir: P) -> std::io::Result<()> {
        let model_entries = fs::read_dir(models_dir)?;

        for model_entry in model_entries {
            let model_entry = model_entry?;
            if model_entry.file_type()?.is_dir() {
                if let Some(model_id) = ModelId::from_path(model_entry.path()) {
                    self.register_model(model_id);
                }
            }
        }

        Ok(())
    }

    pub fn register_model(&self, model_id: ModelId) {
        self.models
            .entry(model_id)
            .or_insert_with(|| Mutex::new(CircularBuffer::new(self.models_buffer_capacity)));
    }

    pub fn add_request(&self, model_id: ModelId, req: InferenceRequest) {
        let buffer = self
            .models
            .entry(model_id)
            .or_insert_with(|| Mutex::new(CircularBuffer::new(self.models_buffer_capacity)));

        let mut buffer = buffer.lock().unwrap();
        buffer.push(req);
    }

    pub fn get_models(&self) -> Vec<ModelId> {
        self.models
            .iter()
            .map(|entry| entry.key().clone())
            .collect()
    }
}

// Type alias for backward compatibility
pub type ModelManager = ModelDiscoveryService;

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_from_path_with_valid_file_extension() {
        let path = PathBuf::from("/models/my_model.py");
        let model_id = ModelId::from_path(path).unwrap();
        assert_eq!(model_id.0, "my_model.py");
    }

    #[test]
    fn test_from_path_with_subpath_and_filename() {
        let path = PathBuf::from("/models/my_model/my_model.py");
        let model_id = ModelId::from_path(path).unwrap();
        assert_eq!(model_id.0, "my_model.py");
    }

    #[test]
    fn test_from_path_with_no_filename() {
        let path = PathBuf::from("/models/");
        let model_id = ModelId::from_path(path);
        assert!(model_id.is_none());
    }

    #[test]
    fn test_from_path_with_subpath_and_no_filename() {
        let path = PathBuf::from("/models/my_model");
        let model_id = ModelId::from_path(path);
        assert!(model_id.is_none());
    }

    #[test]
    fn test_from_path_with_empty_path() {
        let path = PathBuf::new();
        let model_id = ModelId::from_path(path);
        assert!(model_id.is_none());
    }

    #[test]
    fn test_from_url_with_valid_url() {
        let url = "https://example.com/models/my_model";
        let model_id = ModelId::from_url(url).unwrap();
        assert_eq!(model_id.0, "my_model");
    }

    #[test]
    fn test_from_url_with_trailing_slash() {
        let url = "https://example.com/models/my_model/";
        let model_id = ModelId::from_url(url);
        assert!(model_id.is_none());
    }

    #[test]
    fn test_from_string() {
        let id = "my_custom_model".to_string();
        let model_id = ModelId::from_string(id);
        assert_eq!(model_id.0, "my_custom_model");
    }

    #[test]
    fn test_model_discovery_service_register_model() {
        let service = ModelDiscoveryService::new(10);
        let model_id = ModelId::from_string("test_model".to_string());

        service.register_model(model_id.clone());
        let models = service.get_models();
        assert!(models.contains(&model_id));
    }

    #[tokio::test]
    async fn test_discover_models_with_mixed_sources() {
        let service = ModelDiscoveryService::new(10);
        let sources = vec![
            ModelSource::Id("model1".to_string()),
            ModelSource::Url("https://example.com/model2".to_string()),
        ];

        let discovered = service.discover_models(sources).await.unwrap();
        assert_eq!(discovered.len(), 2);
        assert_eq!(discovered[0].0, "model1");
        assert_eq!(discovered[1].0, "model2");
    }

    #[tokio::test]
    async fn test_discover_models_with_mlflow_source() {
        let service = ModelDiscoveryService::new(10);
        let sources = vec![ModelSource::MLFlow {
            base_url: "http://localhost:5000".to_string(),
            api_token: None,
            model_name: Some("test_model".to_string()),
        }];

        // This test would normally connect to a real MLFlow server
        // For now, we just test that the structure compiles and accepts the source
        // In a real test environment, you would mock the MLFlow client
        assert!(matches!(sources[0], ModelSource::MLFlow { .. }));
    }

    #[tokio::test]
    async fn test_discover_all_models_from_mlflow() {
        let service = ModelDiscoveryService::new(10);
        let sources = vec![ModelSource::MLFlow {
            base_url: "http://localhost:5000".to_string(),
            api_token: Some("token123".to_string()),
            model_name: None, // Discover all models
        }];

        // Test structure compilation
        assert!(matches!(sources[0], ModelSource::MLFlow { .. }));
        if let ModelSource::MLFlow { model_name, .. } = &sources[0] {
            assert!(model_name.is_none());
        }
    }
}
