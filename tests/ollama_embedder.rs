use rag::embedder::{Embedder, OllamaEmbedder};

#[tokio::test]
async fn test_ollama_embedder() {
    let embedder = OllamaEmbedder::new("http://localhost:11434", "mxbai-embed-large")
        .await
        .unwrap();

    let result = embedder
        .embed("The sky is blue because of Rayleigh scattering")
        .await;

    assert!(result.is_ok());
}
