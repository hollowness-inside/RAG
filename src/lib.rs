#![allow(async_fn_in_trait)]

pub mod chain;
pub use chain::{RagChain, RagConfig};

pub mod db;
pub mod embedder;
pub mod indexer;

use std::hash::{DefaultHasher, Hash, Hasher};
pub fn calculate_hash<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}
