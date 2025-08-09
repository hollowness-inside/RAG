use ollama_rs::{
    Ollama,
    generation::chat::{ChatMessage, request::ChatMessageRequest},
};
use rag::{FileHashStorage, OllamaEmbedder, QdrantDB, RagIndex, RagResult};

const RAG_PROMPT: &str = include_str!("../rag.prompt");

struct Config {
    ollama_url: String,
    embed_model: String,
    qdrant_url: String,
    collection: String,
    data_dir: String,
    hash_storage: String,
    query: String,
    vector_size: u64,
    text_splitter_chunk: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            ollama_url: "http://localhost:11434".to_string(),
            embed_model: "mxbai-embed-large".to_string(),
            qdrant_url: "http://localhost:6334".to_string(),
            collection: "rag".to_string(),
            data_dir: "data".to_string(),
            hash_storage: "hash.db".to_string(),
            query: "What is the structure of SensorPacket?".to_string(),
            vector_size: 1024,
            text_splitter_chunk: 1024,
        }
    }
}

impl Config {
    fn with_query(query: &str) -> Self {
        Config {
            query: query.to_string(),
            ..Default::default()
        }
    }
}

#[tokio::main]
async fn main() -> RagResult<()> {
    let cfg = Config::with_query("Who is Elara?");

    let embedder = OllamaEmbedder::new(cfg.ollama_url, cfg.embed_model).await?;
    let vector_db = QdrantDB::new(&cfg.qdrant_url, &cfg.collection, cfg.vector_size).await?;
    let hash_storage = FileHashStorage::new(cfg.hash_storage)?;
    let mut indexer = RagIndex::new(embedder, vector_db, hash_storage, cfg.text_splitter_chunk);
    indexer.embed_directory(&cfg.data_dir).await?;

    let mut history: Vec<ChatMessage> = vec![ChatMessage::system(RAG_PROMPT.to_string())]
        .into_iter()
        .chain(
            indexer
                .search_embedding(&cfg.query)
                .await?
                .into_iter()
                .take(5)
                .filter(|p| p.similarity >= 0.4)
                .map(|res| {
                    let mut content = res.content;
                    content.push_str("[SOURCE] ");
                    content.push_str(&res.source);
                    ChatMessage::system(content)
                })
                .collect::<Vec<_>>(),
        )
        .collect();

    let res = Ollama::default()
        .send_chat_messages_with_history(
            &mut history,
            ChatMessageRequest::new(
                "qwen3:latest".into(),
                vec![ChatMessage::user(cfg.query.clone())],
            ),
        )
        .await?
        .message;

    println!("{}", res.content);
    Ok(())
}
