mod error;
pub use error::{RagError, RagResult};

pub mod ollama_embedder;

mod embedder;
pub use embedder::Embedder;
