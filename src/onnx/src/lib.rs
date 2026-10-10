//! ONNX Runtime backend.
//!
//! This is the default backend: it serves any artifact whose platform is
//! `onnx_onnxv1`. It is the only crate in the workspace that knows ONNX Runtime
//! exists; everything else talks to [`galemind_core::Backend`].
//!
//! # Memory
//!
//! Two decisions here decide whether the process grows without bound under
//! load, and both are deliberate:
//!
//! * One session per loaded model version, built once and reused for every
//!   inference. Building a session parses and optimizes the graph and
//!   allocates its weights, so building one per request both costs more than
//!   the inference and leaves the arena to grow behind it.
//! * The CPU memory arena is off by default. The arena never returns memory to
//!   the allocator: it keeps every block it has ever needed, sized for the
//!   largest batch it has seen, per session. With a model per tenant and a
//!   batch size that varies, that reads as an unbounded leak -- a process that
//!   starts at 20 MB and settles at 100 MB after a few thousand requests --
//!   when it is really retained arena. Turning it off costs a few percent of
//!   latency and gives memory back between requests.
//!
//! [`OnnxRuntimeConfig`] can turn the arena back on for a deployment that
//! serves one model with a fixed batch size, where the arena is free
//! throughput.

#![forbid(unsafe_code)]

mod convert;

use convert::{from_ort_value, tensor_metadata, to_ort_value};
use galemind_core::{
    Backend, BackendFactory, InferError, InferRequest, InferResponse, ModelArtifact, ModelMetadata,
    Parameters, PLATFORM_ONNX,
};
use ort::ep;
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

/// How sessions are built: threads, and whether the arena is used.
///
/// The defaults are the ones a multi-model CPU server wants: one thread per
/// operator, so that the parallelism comes from serving models and batches
/// side by side rather than from inside a single small operator, and no arena,
/// so that resident memory follows the work actually in flight.
#[derive(Debug, Clone, Copy)]
pub struct OnnxRuntimeConfig {
    /// Threads ONNX Runtime may use inside a single operator, when the model's
    /// own config does not say. 0 lets ONNX Runtime choose, which means one
    /// thread per core *per session* -- with several models loaded that
    /// oversubscribes the machine badly.
    pub intra_op_threads: usize,
    /// Threads ONNX Runtime may use to run independent operators at once. Only
    /// meaningful with `parallel_execution`; 0 lets ONNX Runtime choose.
    pub inter_op_threads: usize,
    /// Run independent branches of the graph concurrently. Worth it for wide
    /// graphs such as an ensemble, pointless for a deep chain.
    pub parallel_execution: bool,
    /// Keep ONNX Runtime's CPU arena allocator. Off by default; see the module
    /// documentation.
    pub cpu_mem_arena: bool,
    /// Let ONNX Runtime reuse one allocation plan across runs. It assumes the
    /// shapes stay the same, so it is off: batches here vary in size by
    /// design.
    pub memory_pattern: bool,
}

impl Default for OnnxRuntimeConfig {
    fn default() -> Self {
        Self {
            intra_op_threads: 1,
            inter_op_threads: 0,
            parallel_execution: false,
            cpu_mem_arena: false,
            memory_pattern: false,
        }
    }
}

/// Creates [`OnnxBackend`]s. Register it to serve ONNX models.
#[derive(Debug, Default)]
pub struct OnnxBackendFactory {
    config: OnnxRuntimeConfig,
}

impl OnnxBackendFactory {
    /// `intra_op_threads` of 0 lets ONNX Runtime pick; everything else takes
    /// its default.
    pub fn new(default_intra_op_threads: usize) -> Self {
        Self::with_config(OnnxRuntimeConfig {
            intra_op_threads: default_intra_op_threads,
            ..OnnxRuntimeConfig::default()
        })
    }

    pub fn with_config(config: OnnxRuntimeConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &OnnxRuntimeConfig {
        &self.config
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
        // A model may ask for more threads than the server's default, because
        // only the model knows how wide its operators are.
        let config = OnnxRuntimeConfig {
            intra_op_threads: artifact
                .config
                .intra_op_threads
                .unwrap_or(self.config.intra_op_threads),
            ..self.config
        };
        Ok(Arc::new(OnnxBackend::load(artifact, config)?))
    }
}

/// One ONNX model held in memory as a live session.
///
/// The session is built once, when the model is loaded into the cache, and
/// reused for every inference against that version; it is dropped only when
/// the cache unloads the model.
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
    pub fn load(artifact: &ModelArtifact, config: OnnxRuntimeConfig) -> Result<Self> {
        let mut builder = Session::builder()
            .map_err(backend_err)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(backend_err)?
            // Registering the CPU provider explicitly is the only way to reach
            // `Disable/EnableCpuMemArena` through the safe API.
            .with_execution_providers([ep::CPU::default()
                .with_arena_allocator(config.cpu_mem_arena)
                .build()])
            .map_err(backend_err)?
            .with_memory_pattern(config.memory_pattern)
            .map_err(backend_err)?;
        if config.intra_op_threads > 0 {
            builder = builder
                .with_intra_threads(config.intra_op_threads)
                .map_err(backend_err)?;
        }
        if config.parallel_execution {
            builder = builder.with_parallel_execution(true).map_err(backend_err)?;
            if config.inter_op_threads > 0 {
                builder = builder
                    .with_inter_threads(config.inter_op_threads)
                    .map_err(backend_err)?;
            }
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
            intra_op_threads = config.intra_op_threads,
            cpu_mem_arena = config.cpu_mem_arena,
            bytes = artifact.bytes.len(),
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
