use qdrant_client::QdrantError;

pub type RagResult<T> = std::result::Result<T, RagError>;

#[derive(Debug)]
pub enum RagError {
    Response(String),
    Url(String),
    Rewqest(String),
    Serde(String),
    VectorDB(String),
}

impl From<reqwest::Error> for RagError {
    fn from(err: reqwest::Error) -> Self {
        RagError::Rewqest(err.to_string())
    }
}

impl From<serde_json::Error> for RagError {
    fn from(err: serde_json::Error) -> Self {
        RagError::Serde(err.to_string())
    }
}

impl From<QdrantError> for RagError {
    fn from(err: QdrantError) -> Self {
        RagError::VectorDB(err.to_string())
    }
}
