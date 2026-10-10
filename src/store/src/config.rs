//! The optional `config.json` that sits beside a model.

use galemind_core::{InferError, ModelConfig};
use serde::Deserialize;
use std::path::Path;

/// Settings read from `<repository>/<model>/config.json`.
///
/// Every field is optional and unknown fields are ignored, so a config may also
/// carry documentation for humans (a description, feature names, class labels)
/// without the server caring.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct ConfigFile {
    platform: Option<String>,
    max_batch_size: Option<usize>,
    intra_op_threads: Option<usize>,
}

/// Reads a model's config, returning defaults when the file is absent.
pub async fn read_model_config(path: &Path) -> Result<ModelConfig, InferError> {
    let bytes = match tokio::fs::read(path).await {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(ModelConfig::default()),
        Err(e) => {
            return Err(InferError::Store(format!(
                "cannot read model config {}: {e}",
                path.display()
            )))
        }
    };

    let file: ConfigFile = serde_json::from_slice(&bytes)
        .map_err(|e| InferError::Store(format!("invalid model config {}: {e}", path.display())))?;

    if file.max_batch_size == Some(0) {
        return Err(InferError::Store(format!(
            "invalid model config {}: max_batch_size must be at least 1",
            path.display()
        )));
    }

    Ok(ModelConfig {
        platform: file.platform,
        max_batch_size: file.max_batch_size,
        intra_op_threads: file.intra_op_threads,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn config_from(json: &str) -> Result<ModelConfig, InferError> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        tokio::fs::write(&path, json).await.unwrap();
        read_model_config(&path).await
    }

    #[tokio::test]
    async fn a_missing_config_yields_defaults() {
        let config = read_model_config(Path::new("/nonexistent/config.json"))
            .await
            .unwrap();
        assert_eq!(config, ModelConfig::default());
    }

    #[tokio::test]
    async fn documentation_fields_are_ignored() {
        let config =
            config_from(r#"{"max_batch_size": 8, "description": "notes", "features": ["a", "b"]}"#)
                .await
                .unwrap();
        assert_eq!(config.max_batch_size, Some(8));
    }

    #[tokio::test]
    async fn a_zero_batch_size_is_rejected() {
        assert!(config_from(r#"{"max_batch_size": 0}"#).await.is_err());
    }

    #[tokio::test]
    async fn malformed_json_is_rejected() {
        assert!(config_from("{").await.is_err());
    }
}
