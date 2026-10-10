//! The hot tier of the model cache.
//!
//! A loaded model is a live runtime session, which cannot be serialized, so the
//! cache that matters at request time is in memory. This crate holds it: a
//! bounded map of loaded backends, evicted least-recently-used when full and
//! swept when an entry has sat unused past its time-to-live.
//!
//! The cold tier is a [`galemind_core::ModelStore`] (RocksDB in production),
//! which this cache reads on a miss. An optional [`galemind_core::UsageIndex`]
//! records every load so the eviction order is not lost on restart.

#![forbid(unsafe_code)]

mod lru;

pub use lru::{LruTtlCache, LruTtlConfig};
