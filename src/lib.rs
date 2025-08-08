mod error;
pub use error::{RagError, RagResult};

mod embedder;
pub use embedder::{Embedder, OllamaEmbedder};

mod db;
pub use db::{QdrantDB, VectorDB};
