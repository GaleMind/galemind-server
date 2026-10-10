//! Choosing a backend for an artifact.

use crate::error::InferError;
use crate::model::ModelArtifact;
use crate::traits::{Backend, BackendFactory};
use std::sync::Arc;

/// The set of backends this server can serve models with.
///
/// Loading asks each registered factory, in registration order, whether it
/// accepts the artifact. Supporting another runtime is therefore purely
/// additive: register its factory and nothing else changes.
#[derive(Default)]
pub struct BackendRegistry {
    factories: Vec<Arc<dyn BackendFactory>>,
}

impl BackendRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a factory. Earlier registrations win when both accept an artifact.
    pub fn register(&mut self, factory: Arc<dyn BackendFactory>) -> &mut Self {
        self.factories.push(factory);
        self
    }

    pub fn with(mut self, factory: Arc<dyn BackendFactory>) -> Self {
        self.register(factory);
        self
    }

    /// The platform strings this server can serve.
    pub fn platforms(&self) -> Vec<&'static str> {
        self.factories.iter().map(|f| f.platform()).collect()
    }

    pub fn is_empty(&self) -> bool {
        self.factories.is_empty()
    }

    /// Loads an artifact with the first factory that accepts it.
    pub fn load(&self, artifact: &ModelArtifact) -> Result<Arc<dyn Backend>, InferError> {
        self.factories
            .iter()
            .find(|factory| factory.accepts(artifact))
            .ok_or_else(|| {
                InferError::Unsupported(format!(
                    "no backend serves platform '{}' (available: {})",
                    artifact.platform,
                    self.platforms().join(", ")
                ))
            })?
            .load(artifact)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infer::{InferRequest, InferResponse};
    use crate::model::{ModelConfig, ModelKey, ModelMetadata};

    struct StubBackend(ModelMetadata);

    impl Backend for StubBackend {
        fn metadata(&self) -> &ModelMetadata {
            &self.0
        }
        fn infer(&self, _request: &InferRequest) -> Result<InferResponse, InferError> {
            unimplemented!("not needed for registry tests")
        }
        fn size_bytes(&self) -> usize {
            0
        }
    }

    struct StubFactory(&'static str);

    impl BackendFactory for StubFactory {
        fn platform(&self) -> &'static str {
            self.0
        }
        fn accepts(&self, artifact: &ModelArtifact) -> bool {
            artifact.platform == self.0
        }
        fn load(&self, artifact: &ModelArtifact) -> Result<Arc<dyn Backend>, InferError> {
            Ok(Arc::new(StubBackend(ModelMetadata {
                name: artifact.key.name.clone(),
                versions: vec![artifact.key.version.clone()],
                platform: self.0.to_string(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                max_batch_size: None,
            })))
        }
    }

    fn artifact(platform: &str) -> ModelArtifact {
        ModelArtifact {
            key: ModelKey::new("m", "1"),
            platform: platform.to_string(),
            digest: String::new(),
            config: ModelConfig::default(),
            bytes: Vec::new(),
        }
    }

    #[test]
    fn the_factory_matching_the_platform_loads_the_artifact() {
        let registry = BackendRegistry::new()
            .with(Arc::new(StubFactory("onnx_onnxv1")))
            .with(Arc::new(StubFactory("pytorch_torchscript")));

        let backend = registry.load(&artifact("pytorch_torchscript")).unwrap();
        assert_eq!(backend.metadata().platform, "pytorch_torchscript");
        assert_eq!(registry.platforms().len(), 2);
    }

    #[test]
    fn an_unserved_platform_is_reported_with_the_available_ones() {
        let registry = BackendRegistry::new().with(Arc::new(StubFactory("onnx_onnxv1")));
        let Err(err) = registry.load(&artifact("tensorrt_plan")) else {
            panic!("an unserved platform must not load");
        };
        assert!(err.to_string().contains("onnx_onnxv1"), "{err}");
        assert!(err.is_client_error());
    }

    #[test]
    fn an_empty_registry_serves_nothing() {
        assert!(BackendRegistry::new().is_empty());
    }
}
