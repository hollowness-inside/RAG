use ollama_rs::{
    Ollama,
    generation::chat::{ChatMessage, request::ChatMessageRequest},
};
use rag::{FileHashStorage, OllamaEmbedder, QdrantDB, RagIndex, RagResult};

const RAG_PROMPT: &str = include_str!("../rag.prompt");

#[tokio::main]
async fn main() -> RagResult<()> {
    const VECTOR_SIZE: u64 = 1024;
    const TEXT_SPLITTER_CHUNK: usize = 1024;

    const OLLAMA_MODEL: &str = "mxbai-embed-large";
    const OLLAMA_URL: &str = "http://localhost:11434";

    const QDRANT_URL: &str = "http://localhost:6334";
    const QDRANT_COLLECTION: &str = "rag";

    let embedder = OllamaEmbedder::new(OLLAMA_URL, OLLAMA_MODEL).await?;
    let vector_db = QdrantDB::new(QDRANT_URL, QDRANT_COLLECTION, VECTOR_SIZE).await?;
    let hash_storage = FileHashStorage::new("hash.db")?;

    let mut indexer = RagIndex::new(embedder, vector_db, hash_storage, TEXT_SPLITTER_CHUNK);
    indexer.embed_directory("data").await?;

    let response = indexer.search_embedding("SensorPacket").await?;

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
