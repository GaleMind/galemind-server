//! What the repository holds, as the `/v2/models/list` and
//! `/v2/models/info/{name}` endpoints report it.
//!
//! The Predict Protocol v2 has one metadata call, for one model version. It
//! answers "what does this model take and return", and nothing answers "what
//! can this server serve" or "which versions of this model exist, and which is
//! the newest". Those two are what an operator actually asks first, so they
//! are served as extensions under `/v2/models/list` and
//! `/v2/models/info/{name}` rather than left to a reader of the log.
//!
//! Both are built from the registry's view of the repository plus the store's
//! metadata records, which hold a version's platform, digest and settings
//! apart from its bytes. Listing therefore loads nothing: a server with ten
//! models and a cache that holds eight can be listed without evicting
//! anything.

use crate::model::{ModelConfig, ModelKey, TensorMetadata};

/// What a store knows about one model version without reading its bytes.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelDescriptor {
    pub key: ModelKey,
    pub platform: String,
    /// Hex-encoded SHA-256 of the artifact's bytes.
    pub digest: String,
    pub config: ModelConfig,
    /// Size of the artifact, when the store records it.
    pub size_bytes: Option<usize>,
}

/// One model in the catalogue: every version it has, and which is newest.
#[derive(Debug, Clone, PartialEq)]
pub struct CatalogueEntry {
    pub name: String,
    /// Oldest first, in the order the server resolves them.
    pub versions: Vec<String>,
    /// The version an unversioned request resolves to.
    pub latest_version: String,
    pub platform: String,
    /// Versions in memory right now. Loading is lazy, so this is a reading of
    /// the cache rather than a property of the model.
    pub loaded_versions: Vec<String>,
}

impl CatalogueEntry {
    /// Whether the version an unversioned request would use is in memory, so
    /// that a request for it pays no load.
    pub fn latest_is_loaded(&self) -> bool {
        self.loaded_versions.contains(&self.latest_version)
    }
}

/// One version of a model, with as much detail as is available cheaply.
///
/// `inputs` and `outputs` come from the loaded session, so they are present
/// for a version that is in memory, or for any version when the caller asked
/// for the model to be loaded. They are empty rather than guessed otherwise:
/// the signature is a property of the artifact, and only the runtime that
/// opened it can report it.
#[derive(Debug, Clone, PartialEq)]
pub struct VersionInfo {
    pub version: String,
    pub platform: String,
    pub digest: String,
    pub size_bytes: Option<usize>,
    pub max_batch_size: Option<usize>,
    pub intra_op_threads: Option<usize>,
    /// Whether the version is in memory now.
    pub loaded: bool,
    /// Whether an unversioned request resolves to this version.
    pub is_latest: bool,
    pub inputs: Vec<TensorMetadata>,
    pub outputs: Vec<TensorMetadata>,
}

/// Every version of one model.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelInfo {
    pub name: String,
    pub latest_version: String,
    /// Oldest first, matching the order versions are resolved in.
    pub versions: Vec<VersionInfo>,
}

impl ModelInfo {
    /// The newest version's detail, which is what an unversioned request gets.
    pub fn latest(&self) -> Option<&VersionInfo> {
        self.versions.iter().find(|version| version.is_latest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(latest: &str, loaded: &[&str]) -> CatalogueEntry {
        CatalogueEntry {
            name: "energy".to_string(),
            versions: vec!["1".to_string(), "2".to_string()],
            latest_version: latest.to_string(),
            platform: "onnx_onnxv1".to_string(),
            loaded_versions: loaded.iter().map(|v| v.to_string()).collect(),
        }
    }

    #[test]
    fn a_catalogue_entry_reports_whether_the_newest_version_is_in_memory() {
        assert!(entry("2", &["2"]).latest_is_loaded());
        // An older version being loaded does not spare an unversioned request
        // a load, which is the question this answers.
        assert!(!entry("2", &["1"]).latest_is_loaded());
        assert!(!entry("2", &[]).latest_is_loaded());
    }

    #[test]
    fn model_info_finds_the_version_an_unversioned_request_resolves_to() {
        let version = |v: &str, is_latest: bool| VersionInfo {
            version: v.to_string(),
            platform: "onnx_onnxv1".to_string(),
            digest: String::new(),
            size_bytes: None,
            max_batch_size: None,
            intra_op_threads: None,
            loaded: false,
            is_latest,
            inputs: Vec::new(),
            outputs: Vec::new(),
        };
        let info = ModelInfo {
            name: "energy".to_string(),
            latest_version: "2".to_string(),
            versions: vec![version("1", false), version("2", true)],
        };
        assert_eq!(info.latest().map(|v| v.version.as_str()), Some("2"));
    }
}
