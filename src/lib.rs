mod error;
use std::hash::{DefaultHasher, Hash, Hasher};

pub use error::{RagError, RagResult};

mod embedder;
pub use embedder::{Embedder, OllamaEmbedder};

mod db;
pub use db::{QdrantDB, VectorDB};

mod storage;
pub use storage::{FileStorage, Storage};

mod rag;
pub use rag::Rag;

pub fn calculate_hash<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}
