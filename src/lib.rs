mod error;
use std::hash::{DefaultHasher, Hash, Hasher};

pub use error::{RagError, RagResult};

mod embedder;
pub use embedder::{Embedder, OllamaEmbedder};

mod db;
pub use db::{QdrantDB, VectorDB};

mod storage;
pub use storage::{FileHashStorage, HashStorage};

mod embeding_storage;
pub use embeding_storage::EmbeddingStorage;

pub fn calculate_hash<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}
