use crate::RagResult;

pub trait Embedder {
    fn embed(&self, text: &str) -> RagResult<Vec<f32>>;
}
