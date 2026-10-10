//! The filesystem model repository.

use crate::config::read_model_config;
use crate::digest::sha256_hex;
use crate::platform::{candidate_file_names, platform_for_file};
use async_trait::async_trait;
use galemind_core::{InferError, ModelArtifact, ModelDescriptor, ModelKey, ModelStore};
use std::path::{Path, PathBuf};
use tracing::warn;

type Result<T> = std::result::Result<T, InferError>;

/// A model repository laid out as directories:
///
/// ```text
/// models/
///   building_energy/
///     config.json          optional
///     1/model.onnx
///     2/model.onnx
/// ```
///
/// A version directory's name is the model version, so versions are whatever
/// the operator calls them. A model directory holding the model file directly,
/// with no version directory, is served as version `"1"`.
pub struct FsModelStore {
    root: PathBuf,
}

impl FsModelStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The model file inside a version directory, if one is present.
    async fn model_file(dir: &Path) -> Option<PathBuf> {
        for name in candidate_file_names() {
            let candidate = dir.join(name);
            if tokio::fs::try_exists(&candidate).await.unwrap_or(false) {
                return Some(candidate);
            }
        }
        None
    }

    async fn read_dir_names(dir: &Path) -> Result<Vec<String>> {
        let mut entries = tokio::fs::read_dir(dir).await.map_err(|e| {
            InferError::Store(format!(
                "cannot read model repository {}: {e}",
                dir.display()
            ))
        })?;

        let mut names = Vec::new();
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| InferError::Store(format!("cannot walk {}: {e}", dir.display())))?
        {
            let file_type = entry
                .file_type()
                .await
                .map_err(|e| InferError::Store(format!("cannot stat {:?}: {e}", entry.path())))?;
            if !file_type.is_dir() {
                continue;
            }
            match entry.file_name().into_string() {
                Ok(name) if !name.starts_with('.') => names.push(name),
                Ok(_) => {}
                Err(name) => warn!(?name, "skipping directory with a non-UTF-8 name"),
            }
        }
        names.sort();
        Ok(names)
    }

    /// Resolves a key to the model file that holds it.
    async fn locate(&self, key: &ModelKey) -> Result<PathBuf> {
        let versioned = self.root.join(&key.name).join(&key.version);
        if let Some(path) = Self::model_file(&versioned).await {
            return Ok(path);
        }
        if key.version == "1" {
            if let Some(path) = Self::model_file(&self.root.join(&key.name)).await {
                return Ok(path);
            }
        }
        Err(InferError::VersionNotFound {
            name: key.name.clone(),
            version: key.version.clone(),
        })
    }
}

#[async_trait]
impl ModelStore for FsModelStore {
    async fn list(&self) -> Result<Vec<ModelKey>> {
        let mut keys = Vec::new();

        for name in Self::read_dir_names(&self.root).await? {
            let model_dir = self.root.join(&name);

            // An unversioned model directory is version "1".
            if FsModelStore::model_file(&model_dir).await.is_some() {
                keys.push(ModelKey::new(&name, "1"));
                continue;
            }

            for version in Self::read_dir_names(&model_dir).await? {
                if FsModelStore::model_file(&model_dir.join(&version))
                    .await
                    .is_some()
                {
                    keys.push(ModelKey::new(&name, &version));
                } else {
                    warn!(model = %name, %version, "version directory holds no recognised model file");
                }
            }
        }

        keys.sort();
        Ok(keys)
    }

    async fn fetch(&self, key: &ModelKey) -> Result<ModelArtifact> {
        let path = self.locate(key).await?;
        let platform = platform_for_file(&path)
            .ok_or_else(|| {
                InferError::Unsupported(format!("unrecognised model file {}", path.display()))
            })?
            .to_string();

        let bytes = tokio::fs::read(&path)
            .await
            .map_err(|e| InferError::Store(format!("cannot read {}: {e}", path.display())))?;

        let config = read_model_config(&self.root.join(&key.name).join("config.json")).await?;
        let platform = config.platform.clone().unwrap_or(platform);

        Ok(ModelArtifact {
            key: key.clone(),
            platform,
            digest: sha256_hex(&bytes),
            config,
            bytes,
        })
    }

    /// Describes a version by stat-ing its file rather than reading it.
    ///
    /// The digest is left empty: it is a hash of the bytes, and hashing them
    /// means reading them, which is the cost this exists to avoid. A caller
    /// that needs the digest of a filesystem artifact fetches it.
    async fn describe(&self, key: &ModelKey) -> Result<ModelDescriptor> {
        let path = self.locate(key).await?;
        let platform = platform_for_file(&path)
            .ok_or_else(|| {
                InferError::Unsupported(format!("unrecognised model file {}", path.display()))
            })?
            .to_string();
        let size_bytes = tokio::fs::metadata(&path)
            .await
            .map(|metadata| metadata.len() as usize)
            .ok();
        let config = read_model_config(&self.root.join(&key.name).join("config.json")).await?;

        Ok(ModelDescriptor {
            key: key.clone(),
            platform: config.platform.clone().unwrap_or(platform),
            digest: String::new(),
            config,
            size_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use galemind_core::PLATFORM_ONNX;
    use tempfile::TempDir;

    async fn repository() -> TempDir {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        for (model, versions) in [("energy", vec!["1", "2"]), ("bands", vec!["7"])] {
            for version in versions {
                let model_dir = root.join(model).join(version);
                tokio::fs::create_dir_all(&model_dir).await.unwrap();
                tokio::fs::write(model_dir.join("model.onnx"), format!("{model}-{version}"))
                    .await
                    .unwrap();
            }
            tokio::fs::write(
                root.join(model).join("config.json"),
                r#"{"max_batch_size": 4}"#,
            )
            .await
            .unwrap();
        }

        // A model stored without a version directory.
        tokio::fs::create_dir_all(root.join("flat")).await.unwrap();
        tokio::fs::write(root.join("flat/model.onnx"), "flat")
            .await
            .unwrap();
        // Noise that must be ignored.
        tokio::fs::create_dir_all(root.join("empty/3"))
            .await
            .unwrap();
        tokio::fs::write(root.join("README.md"), "docs")
            .await
            .unwrap();

        dir
    }

    #[tokio::test]
    async fn listing_finds_every_version_and_skips_directories_without_a_model() {
        let dir = repository().await;
        let store = FsModelStore::new(dir.path());

        assert_eq!(
            store.list().await.unwrap(),
            vec![
                ModelKey::new("bands", "7"),
                ModelKey::new("energy", "1"),
                ModelKey::new("energy", "2"),
                ModelKey::new("flat", "1"),
            ]
        );
    }

    #[tokio::test]
    async fn fetching_reads_bytes_platform_digest_and_config() {
        let dir = repository().await;
        let store = FsModelStore::new(dir.path());

        let artifact = store.fetch(&ModelKey::new("energy", "2")).await.unwrap();
        assert_eq!(artifact.bytes, b"energy-2");
        assert_eq!(artifact.platform, PLATFORM_ONNX);
        assert_eq!(artifact.config.max_batch_size, Some(4));
        assert_eq!(artifact.digest, sha256_hex(b"energy-2"));
    }

    #[tokio::test]
    async fn an_unversioned_model_is_served_as_version_one() {
        let dir = repository().await;
        let store = FsModelStore::new(dir.path());
        let artifact = store.fetch(&ModelKey::new("flat", "1")).await.unwrap();
        assert_eq!(artifact.bytes, b"flat");
    }

    #[tokio::test]
    async fn fetching_an_unknown_version_reports_version_not_found() {
        let dir = repository().await;
        let store = FsModelStore::new(dir.path());
        let err = store
            .fetch(&ModelKey::new("energy", "9"))
            .await
            .unwrap_err();
        assert!(matches!(err, InferError::VersionNotFound { .. }), "{err}");
    }

    #[tokio::test]
    async fn a_missing_repository_is_a_store_error() {
        let store = FsModelStore::new("/nonexistent/models");
        assert!(store.list().await.is_err());
    }
}
