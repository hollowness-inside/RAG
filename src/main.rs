use ollama_rs::{
    Ollama,
    generation::chat::{ChatMessage, request::ChatMessageRequest},
};
use rag::{EmbeddingStorage, FileHashStorage, OllamaEmbedder, QdrantDB, RagResult};

const RAG_PROMPT: &str = include_str!("../rag.prompt");

#[tokio::main]
async fn main() -> RagResult<()> {
    let embedder = OllamaEmbedder::new("http://localhost:11434", "mxbai-embed-large").await?;

    let vector_db = QdrantDB::new("http://localhost:6334", "rag", 1024).await?;

    let hash_storage = FileHashStorage::new("hash.db").unwrap();

    let mut embedder_storage = EmbeddingStorage::new(embedder, vector_db, hash_storage, 1024);
    embedder_storage.embed_directory("data").await?;

    let response = embedder_storage.search_embedding("SensorPacket").await?;

    let mut ollama = Ollama::default();

    let mut history = vec![ChatMessage::system(RAG_PROMPT.to_string())];

    for res in response {
        let mut content = res.content;
        content.push_str("[SOURCE] ");
        content.push_str(&res.source);
        history.push(ChatMessage::assistant(content))
    }

    let res = ollama
        .send_chat_messages_with_history(
            &mut history,
            ChatMessageRequest::new(
                "qwen3:latest".into(),
                vec![ChatMessage::user(
                    "What is the structure of SensorPacket?".into(),
                )],
            ),
        )
        .await?;

    println!("{}", res.message.content);
    Ok(())
}
