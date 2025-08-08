use rag::{Embedder, ollama_embedder::OllamaEmbedder};

#[test]
fn test_ollama_embedder() {
    let embedder = OllamaEmbedder::new("http://localhost:11434", "mxbai-embed-large").unwrap();
    let result = embedder.embed("The sky is blue because of Rayleigh scattering");
    assert!(result.is_ok());
}
