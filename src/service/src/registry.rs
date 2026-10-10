//! Which models exist, and which version a request means.

use galemind_core::{InferError, ModelKey, ModelRef, ModelStore};
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

type Result<T> = std::result::Result<T, InferError>;

/// A snapshot of the model repository: every model name and its versions.
///
/// Refreshing is explicit rather than per request, so serving a request never
/// waits on the store to be listed.
pub struct ModelRegistry {
    store: Arc<dyn ModelStore>,
    models: RwLock<BTreeMap<String, Vec<String>>>,
}

impl ModelRegistry {
    pub fn new(store: Arc<dyn ModelStore>) -> Self {
        Self {
            store,
            models: RwLock::new(BTreeMap::new()),
        }
    }

    /// Re-reads the repository. Returns how many model versions are known.
    pub async fn refresh(&self) -> Result<usize> {
        let keys = self.store.list().await?;
        let mut models: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for key in keys {
            models.entry(key.name).or_default().push(key.version);
        }
        for versions in models.values_mut() {
            versions.sort_by(|a, b| compare_versions(a, b));
        }

        let total = models.values().map(Vec::len).sum();
        info!(
            models = models.len(),
            versions = total,
            "refreshed model registry"
        );
        *self.models.write().await = models;
        Ok(total)
    }

    /// Every known model version, sorted.
    pub async fn keys(&self) -> Vec<ModelKey> {
        let models = self.models.read().await;
        models
            .iter()
            .flat_map(|(name, versions)| versions.iter().map(move |v| ModelKey::new(name, v)))
            .collect()
    }

    /// Every model and its versions, oldest version first.
    pub async fn models(&self) -> Vec<(String, Vec<String>)> {
        self.models
            .read()
            .await
            .iter()
            .map(|(name, versions)| (name.clone(), versions.clone()))
            .collect()
    }

    /// Versions of one model, oldest first.
    pub async fn versions(&self, name: &str) -> Option<Vec<String>> {
        self.models.read().await.get(name).cloned()
    }

    pub async fn is_empty(&self) -> bool {
        self.models.read().await.is_empty()
    }

    /// Resolves a reference to an exact model version.
    ///
    /// A reference without a version selects the newest: versions that look
    /// numeric are compared as numbers, so `10` is newer than `9`; anything
    /// else falls back to string order.
    pub async fn resolve(&self, model: &ModelRef) -> Result<ModelKey> {
        let models = self.models.read().await;
        let versions = models
            .get(&model.name)
            .ok_or_else(|| InferError::ModelNotFound(model.name.clone()))?;

        match &model.version {
            Some(version) => {
                if versions.iter().any(|known| known == version) {
                    Ok(ModelKey::new(&model.name, version))
                } else {
                    Err(InferError::VersionNotFound {
                        name: model.name.clone(),
                        version: version.clone(),
                    })
                }
            }
            None => versions
                .last()
                .map(|version| ModelKey::new(&model.name, version))
                .ok_or_else(|| InferError::ModelNotFound(model.name.clone())),
        }
    }
}

/// Orders two version labels, numerically when both are numbers.
fn compare_versions(a: &str, b: &str) -> std::cmp::Ordering {
    match (a.parse::<u64>(), b.parse::<u64>()) {
        (Ok(a), Ok(b)) => a.cmp(&b),
        _ => a.cmp(b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use galemind_core::{ModelArtifact, ModelConfig, ModelWriter, PLATFORM_ONNX};
    use galemind_store::MemoryModelStore;

    async fn registry(keys: &[(&str, &str)]) -> ModelRegistry {
        let store = Arc::new(MemoryModelStore::new());
        for (name, version) in keys {
            store
                .put(&ModelArtifact {
                    key: ModelKey::new(*name, *version),
                    platform: PLATFORM_ONNX.to_string(),
                    digest: String::new(),
                    config: ModelConfig::default(),
                    bytes: Vec::new(),
                })
                .await
                .unwrap();
        }
        let registry = ModelRegistry::new(store);
        registry.refresh().await.unwrap();
        registry
    }

    #[tokio::test]
    async fn an_unversioned_reference_selects_the_newest_version_numerically() {
        let registry = registry(&[("energy", "2"), ("energy", "10"), ("energy", "9")]).await;
        assert_eq!(
            registry.resolve(&ModelRef::latest("energy")).await.unwrap(),
            ModelKey::new("energy", "10")
        );
        assert_eq!(
            registry.versions("energy").await.unwrap(),
            vec!["2", "9", "10"]
        );
    }

    #[tokio::test]
    async fn non_numeric_versions_fall_back_to_string_order() {
        let registry = registry(&[("energy", "beta"), ("energy", "alpha")]).await;
        assert_eq!(
            registry.resolve(&ModelRef::latest("energy")).await.unwrap(),
            ModelKey::new("energy", "beta")
        );
    }

    #[tokio::test]
    async fn an_explicit_version_is_honoured_and_checked() {
        let registry = registry(&[("energy", "1"), ("energy", "2")]).await;
        assert_eq!(
            registry
                .resolve(&ModelRef::new("energy", Some("1".into())))
                .await
                .unwrap(),
            ModelKey::new("energy", "1")
        );
        let err = registry
            .resolve(&ModelRef::new("energy", Some("7".into())))
            .await
            .unwrap_err();
        assert!(matches!(err, InferError::VersionNotFound { .. }), "{err}");
    }

    #[tokio::test]
    async fn an_unknown_model_is_reported_as_not_found() {
        let registry = registry(&[("energy", "1")]).await;
        let err = registry
            .resolve(&ModelRef::latest("ghost"))
            .await
            .unwrap_err();
        assert!(matches!(err, InferError::ModelNotFound(_)), "{err}");
    }

    #[tokio::test]
    async fn an_empty_repository_lists_nothing() {
        let registry = registry(&[]).await;
        assert!(registry.is_empty().await);
        assert!(registry.keys().await.is_empty());
    }

    #[tokio::test]
    async fn keys_are_listed_for_every_model_and_version() {
        let registry = registry(&[("bands", "1"), ("energy", "1"), ("energy", "2")]).await;
        assert_eq!(
            registry.keys().await,
            vec![
                ModelKey::new("bands", "1"),
                ModelKey::new("energy", "1"),
                ModelKey::new("energy", "2"),
            ]
        );
    }
}
