use ollama_rs::error::OllamaError;
use pdf_extract::OutputError;
use qdrant_client::QdrantError;
use reqwest::header::InvalidHeaderValue;

pub type RagResult<T> = std::result::Result<T, RagError>;

#[derive(Debug)]
pub enum RagError {
    Io(String),
    Response(String),
    Url(String),
    Serde(String),
    VectorDB(String),
    PdfExtract(String),
    Ollama(String),
    Payload(String),
    Request(String),
}

impl From<reqwest::Error> for RagError {
    fn from(err: reqwest::Error) -> Self {
        RagError::Request(err.to_string())
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

impl From<std::io::Error> for RagError {
    fn from(err: std::io::Error) -> Self {
        RagError::Io(err.to_string())
    }
}

impl From<pdf_extract::Error> for RagError {
    fn from(err: pdf_extract::Error) -> Self {
        RagError::PdfExtract(err.to_string())
    }
}

impl From<OutputError> for RagError {
    fn from(err: OutputError) -> Self {
        RagError::PdfExtract(err.to_string())
    }
}

impl From<OllamaError> for RagError {
    fn from(err: OllamaError) -> Self {
        RagError::Ollama(err.to_string())
    }
}

impl From<InvalidHeaderValue> for RagError {
    fn from(err: InvalidHeaderValue) -> Self {
        RagError::Request(err.to_string())
    }
}
