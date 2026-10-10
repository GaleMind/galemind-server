//! Model repository stores.
//!
//! A store answers two questions: which model versions exist, and what are the
//! bytes of one of them. [`FsModelStore`] reads a directory tree;
//! [`RocksModelStore`] persists artifacts, metadata and usage records in
//! RocksDB; [`MemoryModelStore`] keeps everything in memory.
//!
//! The server normally ingests the filesystem repository into RocksDB at
//! startup (see [`ingest`]) and serves from RocksDB afterwards, so model bytes
//! have one durable home and the cache's eviction order survives a restart.

#![forbid(unsafe_code)]

mod config;
mod digest;
mod fs;
mod ingest;
mod memory;
mod platform;
#[cfg(feature = "rocksdb-store")]
mod rocks;

pub use config::read_model_config;
pub use digest::sha256_hex;
pub use fs::FsModelStore;
pub use ingest::{ingest, IngestReport};
pub use memory::MemoryModelStore;
pub use platform::{candidate_file_names, platform_for_file};
#[cfg(feature = "rocksdb-store")]
pub use rocks::RocksModelStore;
