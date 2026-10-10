//! Domain model for the GaleMind inference server.
//!
//! This crate holds the vocabulary of the Predict Protocol v2 — datatypes,
//! tensors, model identity, requests, responses — and the traits that every
//! replaceable part of the server implements. It performs no I/O and depends on
//! no transport, runtime or storage engine, so every other crate can depend on
//! it without inheriting those choices.

#![forbid(unsafe_code)]

pub mod clock;
pub mod datatype;
pub mod error;
pub mod infer;
pub mod model;
pub mod registry;
pub mod tensor;
pub mod traits;

pub use clock::{SystemClock, TestClock};
pub use datatype::DataType;
pub use error::InferError;
pub use infer::{InferRequest, InferResponse, Parameter, Parameters, RequestedOutput};
pub use model::{ModelArtifact, ModelConfig, ModelKey, ModelMetadata, ModelRef, TensorMetadata};
pub use registry::BackendRegistry;
pub use tensor::{Tensor, TensorData};
pub use traits::{
    Backend, BackendFactory, CacheStats, Clock, Inference, ModelCache, ModelStore, ModelUsage,
    ModelWriter, UsageIndex,
};

/// The platform string for models served by ONNX Runtime, as named by the
/// Predict Protocol v2.
pub const PLATFORM_ONNX: &str = "onnx_onnxv1";

/// Name reported by the server metadata endpoints.
pub const SERVER_NAME: &str = "galemind";

/// Version reported by the server metadata endpoints.
pub const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");
