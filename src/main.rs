use rag::{FileStorage, OllamaEmbedder, QdrantDB, Rag};

#[tokio::main]
async fn main() {
    let embedder = OllamaEmbedder::new("http://localhost:11434", "mxbai-embed-large")
        .await
        .unwrap();

    let vector_db = QdrantDB::new("http://localhost:6334", "rag", 1024)
        .await
        .unwrap();

    let storage = FileStorage::new("hash.db");

    let mut rag = Rag::new(embedder, vector_db, storage, 700);
    rag.embed_directory("data").await.unwrap();

    let response = rag.search_embedding("Maria Garcia").await.unwrap();
    println!("{:#?}", response);
}
