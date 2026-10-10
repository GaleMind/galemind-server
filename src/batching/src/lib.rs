//! Adaptive request batching.
//!
//! Most models answer a batch of rows in little more time than a single row, so
//! grouping requests raises throughput sharply. Batching on the server keeps
//! that benefit without asking clients to coordinate, and it is the server that
//! can see the whole arrival pattern.
//!
//! One [`Dispatcher`] per model collects requests on a bounded queue and runs
//! them as one call. [`AdaptiveBatchPolicy`] sizes those batches from the
//! traffic actually arriving: quiet periods produce near-immediate batches of
//! one, while load grows both the batch and the window up to their ceilings.
//! The order of requests within a batch is not guaranteed.

#![forbid(unsafe_code)]

mod dispatcher;
mod merge;
mod policy;
mod samples;

pub use dispatcher::{Dispatcher, DispatcherConfig, DispatcherMetrics, DispatcherSnapshot};
pub use merge::{merge_requests, split_response};
pub use policy::{
    AdaptiveBatchConfig, AdaptiveBatchPolicy, BatchPolicy, FixedBatchPolicy, TrafficMeter,
    TrafficStats,
};
pub use samples::RecentSamples;
