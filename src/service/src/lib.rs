//! The orchestration layer.
//!
//! [`InferenceService`] is the only thing the REST and gRPC servers talk to. It
//! resolves a model reference to a version, loads it through the cache, checks
//! the request against the model's real metadata, and routes it to that
//! model's batching dispatcher. The transports stay thin: they decode, call,
//! and encode.

#![forbid(unsafe_code)]

mod builder;
mod listener;
mod metrics;
mod registry;
mod resources;
mod runtime;
mod service;

pub use builder::{ingest_repository, ServiceBuilder};
pub use listener::{bind_with_backlog, DEFAULT_LISTEN_BACKLOG};
pub use metrics::{
    render_prometheus, FailureReason, MetricsRegistry, ModelMetricsSnapshot, PrometheusReport,
};
pub use registry::ModelRegistry;
pub use resources::{ResourceSampler, ResourceSnapshot};
pub use runtime::{platform, RuntimeInfo};
pub use service::InferenceService;
