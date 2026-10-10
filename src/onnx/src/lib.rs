//! ONNX Runtime backend.
//!
//! This is the default backend: it serves any artifact whose platform is
//! `onnx_onnxv1`. It is the only crate in the workspace that knows ONNX Runtime
//! exists; everything else talks to [`galemind_core::Backend`].

#![forbid(unsafe_code)]

mod convert;

use convert::{from_ort_value, tensor_metadata, to_ort_value};
use galemind_core::{
    Backend, BackendFactory, InferError, InferRequest, InferResponse, ModelArtifact, ModelMetadata,
    Parameters, PLATFORM_ONNX,
};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use std::sync::{Arc, Mutex};
use tracing::debug;

/// ONNX Runtime errors carry the type of the call that produced them; this
/// flattens any of them into the domain error.
fn backend_err<T>(e: ort::Error<T>) -> InferError {
    InferError::Backend(e.to_string())
}

type Result<T> = std::result::Result<T, InferError>;

/// Creates [`OnnxBackend`]s. Register it to serve ONNX models.
#[derive(Debug, Default)]
pub struct OnnxBackendFactory {
    /// Threads ONNX Runtime may use inside a single operator, when the model's
    /// own config does not say.
    default_intra_op_threads: usize,
}

impl OnnxBackendFactory {
    /// `intra_op_threads` of 0 lets ONNX Runtime pick.
    pub fn new(default_intra_op_threads: usize) -> Self {
        Self {
            default_intra_op_threads,
        }
    }
}

impl BackendFactory for OnnxBackendFactory {
    fn platform(&self) -> &'static str {
        PLATFORM_ONNX
    }

    fn accepts(&self, artifact: &ModelArtifact) -> bool {
        artifact.platform == PLATFORM_ONNX
    }

    fn load(&self, artifact: &ModelArtifact) -> Result<Arc<dyn Backend>> {
        let threads = artifact
            .config
            .intra_op_threads
            .unwrap_or(self.default_intra_op_threads);
        Ok(Arc::new(OnnxBackend::load(artifact, threads)?))
    }
}

/// One ONNX model held in memory as a live session.
///
/// `Session::run` needs exclusive access, so the session sits behind a mutex.
/// Requests for a single model are already serialized by its batching
/// dispatcher, so this lock is not a contention point in normal operation; the
/// throughput comes from batching rows into one call rather than from
/// concurrent calls.
pub struct OnnxBackend {
    session: Mutex<Session>,
    metadata: ModelMetadata,
    size_bytes: usize,
}

impl OnnxBackend {
    /// Builds a session from an artifact's bytes.
    pub fn load(artifact: &ModelArtifact, intra_op_threads: usize) -> Result<Self> {
        let mut builder = Session::builder()
            .map_err(backend_err)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(backend_err)?;
        if intra_op_threads > 0 {
            builder = builder
                .with_intra_threads(intra_op_threads)
                .map_err(backend_err)?;
        }
        let session = builder
            .commit_from_memory(&artifact.bytes)
            .map_err(backend_err)?;

        let inputs = session
            .inputs()
            .iter()
            .map(|outlet| tensor_metadata(outlet.name(), outlet.dtype()))
            .collect::<Result<Vec<_>>>()?;
        let outputs = session
            .outputs()
            .iter()
            .map(|outlet| tensor_metadata(outlet.name(), outlet.dtype()))
            .collect::<Result<Vec<_>>>()?;

        let metadata = ModelMetadata {
            name: artifact.key.name.clone(),
            versions: vec![artifact.key.version.clone()],
            platform: PLATFORM_ONNX.to_string(),
            inputs,
            outputs,
            max_batch_size: artifact.config.max_batch_size,
        };

        debug!(
            model = %artifact.key,
            inputs = metadata.inputs.len(),
            outputs = metadata.outputs.len(),
            batchable = metadata.is_batchable(),
            "loaded ONNX model"
        );

        Ok(Self {
            session: Mutex::new(session),
            metadata,
            size_bytes: artifact.bytes.len(),
        })
    }
}

impl Backend for OnnxBackend {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn infer(&self, request: &InferRequest) -> Result<InferResponse> {
        let values = request
            .inputs
            .iter()
            .map(|tensor| Ok((tensor.name.clone(), to_ort_value(tensor)?)))
            .collect::<Result<Vec<_>>>()?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| InferError::Internal("ONNX session lock was poisoned".to_string()))?;

        let outputs = session
            .run(values)
            .map_err(|e| InferError::Backend(e.to_string()))?;

        let mut tensors = Vec::with_capacity(self.metadata.outputs.len());
        for declared in &self.metadata.outputs {
            let value = outputs.get(declared.name.as_str()).ok_or_else(|| {
                InferError::Backend(format!(
                    "ONNX Runtime did not return declared output '{}'",
                    declared.name
                ))
            })?;
            tensors.push(from_ort_value(&declared.name, value)?);
        }

        Ok(InferResponse {
            model_name: self.metadata.name.clone(),
            model_version: self.metadata.versions.first().cloned().unwrap_or_default(),
            id: request.id.clone(),
            parameters: Parameters::new(),
            outputs: tensors,
        })
    }

    fn size_bytes(&self) -> usize {
        self.size_bytes
    }
}
