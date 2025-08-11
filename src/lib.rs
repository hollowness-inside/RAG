#![allow(async_fn_in_trait)]

use std::hash::{DefaultHasher, Hash, Hasher};

mod embedder;
pub use embedder::{Embedder, OllamaEmbedder};

mod db;
pub use db::{QdrantDB, RetrievedChunk, VectorDB};

mod indexer;
pub use indexer::RagIndex;

mod chain;
pub use chain::{RagChain, RagConfig};

pub fn calculate_hash<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}
