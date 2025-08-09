use ollama_rs::{
    Ollama,
    generation::chat::{ChatMessage, request::ChatMessageRequest},
};

use crate::{
    Embedder, FileHashStorage, HashStorage, OllamaEmbedder, QdrantDB, RagIndex, RagResult,
    RetrievedChunk, VectorDB,
};

const RAG_PROMPT: &str = include_str!("../rag.prompt");

#[derive(Debug, Clone)]
pub struct RagConfig {
    pub ollama_url: String,

    pub embed_model: String,
    pub ai_model: String,

    pub qdrant_url: String,
    pub collection: String,

    pub hash_storage: String,

    pub top_k: usize,
    pub min_similarity: f32,
    pub vector_size: u64,
    pub text_splitter_chunk: usize,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            ollama_url: "http://localhost:11434".to_string(),

            embed_model: "mxbai-embed-large".to_string(),
            ai_model: "qwen3:latest".to_string(),

            qdrant_url: "http://localhost:6334".to_string(),
            collection: "rag".to_string(),

            hash_storage: "hash.db".to_string(),

            top_k: 5,
            min_similarity: 0.4,
            vector_size: 1024,
            text_splitter_chunk: 1024,
        }
    }
}

pub struct RagChain<E: Embedder, V: VectorDB, H: HashStorage> {
    indexer: RagIndex<E, V, H>,
    config: RagConfig,
}

impl RagChain<OllamaEmbedder, QdrantDB, FileHashStorage> {
    pub async fn new() -> RagResult<Self> {
        Self::with_config(RagConfig::default()).await
    }

    pub async fn with_config(config: RagConfig) -> RagResult<Self> {
        let embedder = OllamaEmbedder::new(&config.ollama_url, &config.embed_model).await?;
        let vectordb =
            QdrantDB::new(&config.qdrant_url, &config.collection, config.vector_size).await?;
        let hash_storage = FileHashStorage::new(&config.hash_storage)?;
        let indexer = RagIndex::new(embedder, vectordb, hash_storage, config.text_splitter_chunk);
        Ok(Self { indexer, config })
    }

    pub async fn embed_directory(&mut self, dir: &str) -> RagResult<()> {
        self.indexer.embed_directory(dir).await
    }

    pub async fn ask(&mut self, query: &str) -> RagResult<ChatMessage> {
        let chunks = self.indexer.search(query).await?;
        let mut history = self.build_chat_history(&chunks, RAG_PROMPT).await?;
        let response = self.rag_request(query, &mut history).await?;
        Ok(response)
    }

    pub async fn rag_request(
        &self,
        query: &str,
        history: &mut Vec<ChatMessage>,
    ) -> RagResult<ChatMessage> {
        let resp = Ollama::default()
            .send_chat_messages_with_history(
                history,
                ChatMessageRequest::new(
                    self.config.ai_model.clone(),
                    vec![ChatMessage::user(query.into())],
                ),
            )
            .await?
            .message;

        Ok(resp)
    }

    async fn build_chat_history(
        &self,
        chunks: &[RetrievedChunk],
        base_prompt: &str,
    ) -> RagResult<Vec<ChatMessage>> {
        const CHUNK_START: &str = "=== DOCUMENT CHUNK START ===\n";
        const CHUNK_END: &str = "\n=== DOCUMENT CHUNK END ===";
        const SOURCE: &str = "\n[SOURCE] ";

        let retrieved = chunks
            .iter()
            .take(self.config.top_k)
            .filter(|p| p.similarity >= self.config.min_similarity)
            .map(|res| {
                let mut content = String::with_capacity(
                    res.content.len()
                        + res.source.len()
                        + CHUNK_START.len()
                        + CHUNK_END.len()
                        + SOURCE.len(),
                );

                content.push_str(CHUNK_START);
                content.push_str(&res.content);
                content.push_str(SOURCE);
                content.push_str(&res.source);
                content.push_str(CHUNK_END);
                ChatMessage::system(content)
            });

        let mut history = Vec::with_capacity(1 + self.config.top_k);
        history.push(ChatMessage::system(base_prompt.to_string()));
        history.extend(retrieved);
        Ok(history)
    }
}
