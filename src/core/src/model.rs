//! Model identity, metadata and on-disk artifacts.

use crate::datatype::DataType;
use crate::error::InferError;
use std::fmt;

/// How a client refers to a model: a name, and optionally a version.
///
/// A request without a version lets the server pick one by its own policy,
/// as the Predict Protocol v2 allows.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelRef {
    pub name: String,
    pub version: Option<String>,
}

impl ModelRef {
    pub fn new(name: impl Into<String>, version: Option<String>) -> Self {
        Self {
            name: name.into(),
            version: version.filter(|v| !v.is_empty()),
        }
    }

    pub fn latest(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: None,
        }
    }
}

impl fmt::Display for ModelRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.version {
            Some(v) => write!(f, "{}/{v}", self.name),
            None => write!(f, "{}", self.name),
        }
    }
}

/// A resolved model identity. Unlike [`ModelRef`], the version is always known,
/// which is what makes it usable as a cache and dispatcher key.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModelKey {
    pub name: String,
    pub version: String,
}

impl ModelKey {
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
        }
    }

    /// The flat form used as a key in the model store.
    pub fn storage_key(&self) -> String {
        format!("{}/{}", self.name, self.version)
    }

    /// Parses the form produced by [`Self::storage_key`].
    pub fn from_storage_key(key: &str) -> Result<Self, InferError> {
        match key.rsplit_once('/') {
            Some((name, version)) if !name.is_empty() && !version.is_empty() => {
                Ok(Self::new(name, version))
            }
            _ => Err(InferError::Store(format!("malformed model key '{key}'"))),
        }
    }
}

impl fmt::Display for ModelKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.name, self.version)
    }
}

/// Shape and type of one model input or output.
///
/// A variable-size dimension is `-1`, as the protocol specifies.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorMetadata {
    pub name: String,
    pub datatype: DataType,
    pub shape: Vec<i64>,
}

impl TensorMetadata {
    /// `true` when the leading dimension is variable, i.e. the tensor can carry
    /// a batch of independent rows.
    pub fn has_dynamic_batch_dim(&self) -> bool {
        matches!(self.shape.first(), Some(&dim) if dim < 0)
    }
}

/// Everything the metadata endpoints report about a model.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelMetadata {
    pub name: String,
    pub versions: Vec<String>,
    /// One of the platform strings named by the protocol, e.g. `"onnx_onnxv1"`.
    pub platform: String,
    pub inputs: Vec<TensorMetadata>,
    pub outputs: Vec<TensorMetadata>,
    /// Largest batch the backend will accept in one call, or `None` when the
    /// model cannot be batched at all.
    pub max_batch_size: Option<usize>,
}

impl ModelMetadata {
    /// `true` when independent requests may be merged along the leading
    /// dimension. Requires every input and output to carry a dynamic batch dim.
    pub fn is_batchable(&self) -> bool {
        self.max_batch_size.is_some_and(|n| n > 1)
            && !self.inputs.is_empty()
            && self
                .inputs
                .iter()
                .all(TensorMetadata::has_dynamic_batch_dim)
            && self
                .outputs
                .iter()
                .all(TensorMetadata::has_dynamic_batch_dim)
    }

    pub fn input(&self, name: &str) -> Option<&TensorMetadata> {
        self.inputs.iter().find(|t| t.name == name)
    }

    pub fn output(&self, name: &str) -> Option<&TensorMetadata> {
        self.outputs.iter().find(|t| t.name == name)
    }
}

/// Per-model settings read from `config.json` beside the model file.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelConfig {
    /// Overrides the platform inferred from the artifact's file extension.
    pub platform: Option<String>,
    pub max_batch_size: Option<usize>,
    /// Threads ONNX Runtime may use inside one operator.
    pub intra_op_threads: Option<usize>,
}

/// A model's serialized bytes together with the identity and settings needed to
/// turn them into a running backend.
#[derive(Clone)]
pub struct ModelArtifact {
    pub key: ModelKey,
    /// Platform string, derived from the artifact or the model config.
    pub platform: String,
    pub bytes: Vec<u8>,
    /// Hex-encoded SHA-256 of `bytes`, used to detect changed artifacts.
    pub digest: String,
    pub config: ModelConfig,
}

impl fmt::Debug for ModelArtifact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelArtifact")
            .field("key", &self.key)
            .field("platform", &self.platform)
            .field("bytes", &format_args!("{} bytes", self.bytes.len()))
            .field("digest", &self.digest)
            .field("config", &self.config)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(name: &str, shape: Vec<i64>) -> TensorMetadata {
        TensorMetadata {
            name: name.to_string(),
            datatype: DataType::Fp32,
            shape,
        }
    }

    #[test]
    fn storage_keys_round_trip_even_when_the_name_contains_a_slash() {
        let key = ModelKey::new("team/energy", "3");
        assert_eq!(key.storage_key(), "team/energy/3");
        assert_eq!(ModelKey::from_storage_key("team/energy/3").unwrap(), key);
    }

    #[test]
    fn a_malformed_storage_key_is_rejected() {
        assert!(ModelKey::from_storage_key("energy").is_err());
        assert!(ModelKey::from_storage_key("energy/").is_err());
    }

    #[test]
    fn an_empty_version_means_no_version() {
        assert_eq!(ModelRef::new("m", Some(String::new())).version, None);
    }

    #[test]
    fn a_model_is_batchable_only_when_every_tensor_has_a_dynamic_batch_dim() {
        let batchable = ModelMetadata {
            name: "m".into(),
            versions: vec!["1".into()],
            platform: "onnx_onnxv1".into(),
            inputs: vec![tensor("in", vec![-1, 8])],
            outputs: vec![tensor("out", vec![-1, 1])],
            max_batch_size: Some(16),
        };
        assert!(batchable.is_batchable());

        let fixed_output = ModelMetadata {
            outputs: vec![tensor("out", vec![1, 1])],
            ..batchable.clone()
        };
        assert!(!fixed_output.is_batchable());

        let unbatchable = ModelMetadata {
            max_batch_size: Some(1),
            ..batchable
        };
        assert!(!unbatchable.is_batchable());
    }
}
