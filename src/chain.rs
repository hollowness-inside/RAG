use anyhow::Result;
use ollama_rs::{
    Ollama,
    generation::chat::{ChatMessage, request::ChatMessageRequest},
};

use crate::{
    db::{QdrantDB, RetrievedChunk, VectorDB},
    embedder::{Embedder, OllamaEmbedder},
    indexer::RagIndex,
};

const RAG_PROMPT: &str = include_str!("../rag.prompt");

#[derive(Debug, Clone)]
pub struct RagBuilder {
    pub ollama_url: String,

    pub embed_model: String,
    pub ai_model: String,

    pub qdrant_url: String,
    pub collection: String,

    pub top_k: usize,
    pub min_similarity: f32,
    pub vector_size: u64,
    pub text_splitter_chunk: usize,
}

impl Default for RagBuilder {
    fn default() -> Self {
        Self {
            ollama_url: "http://localhost:11434".to_string(),

            embed_model: "mxbai-embed-large".to_string(),
            ai_model: "qwen3:latest".to_string(),

            qdrant_url: "http://localhost:6334".to_string(),
            collection: "rag".to_string(),

            top_k: 5,
            min_similarity: 0.4,
            vector_size: 1024,
            text_splitter_chunk: 1024,
        }
    }
}

impl RagBuilder {
    // Builds a RagChain with OllamaEmbedder and QdrantDB as default components.
    pub async fn build_default(self) -> Result<RagChain<OllamaEmbedder, QdrantDB>> {
        let embedder = OllamaEmbedder::new(&self.ollama_url, &self.embed_model).await?;
        let vectordb = QdrantDB::new(&self.qdrant_url, &self.collection, self.vector_size).await?;
        let indexer = RagIndex::new(embedder, vectordb, self.text_splitter_chunk);

        Ok(RagChain {
            indexer,
            ai_model: self.ai_model,
            top_k: self.top_k,
            min_similarity: self.min_similarity,
        })
    }

    pub fn set_ollama_url(mut self, ollama_url: String) -> Self {
        self.ollama_url = ollama_url;
        self
    }

    pub fn set_embed_model(mut self, embed_model: String) -> Self {
        self.embed_model = embed_model;
        self
    }

    pub fn set_ai_model(mut self, ai_model: String) -> Self {
        self.ai_model = ai_model;
        self
    }

    pub fn set_qdrant_url(mut self, qdrant_url: String) -> Self {
        self.qdrant_url = qdrant_url;
        self
    }

    pub fn set_collection(mut self, collection: String) -> Self {
        self.collection = collection;
        self
    }

    pub fn set_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    pub fn set_min_similarity(mut self, min_similarity: f32) -> Self {
        self.min_similarity = min_similarity;
        self
    }

    pub fn set_vector_size(mut self, vector_size: u64) -> Self {
        self.vector_size = vector_size;
        self
    }

    pub fn set_text_splitter_chunk(mut self, text_splitter_chunk: usize) -> Self {
        self.text_splitter_chunk = text_splitter_chunk;
        self
    }
}

pub struct RagChain<E: Embedder, V: VectorDB> {
    indexer: RagIndex<E, V>,

    ai_model: String,
    top_k: usize,
    min_similarity: f32,
}

impl RagChain<OllamaEmbedder, QdrantDB> {
    pub fn builder() -> RagBuilder {
        RagBuilder::default()
    }

    pub async fn embed_directory(&mut self, dir: &str) -> Result<()> {
        self.indexer.embed_directory(dir).await
    }

    pub async fn ask(&mut self, prompt: &str) -> Result<ChatMessage> {
        let chunks: Vec<_> = self.indexer.search(prompt).await?;
        let mut history = self.build_chat_history(&chunks, RAG_PROMPT).await?;
        let response = self.rag_request(prompt, &mut history).await?;
        Ok(response)
    }

    pub async fn rag_request(
        &self,
        query: &str,
        history: &mut Vec<ChatMessage>,
    ) -> Result<ChatMessage> {
        let resp = Ollama::default()
            .send_chat_messages_with_history(
                history,
                ChatMessageRequest::new(
                    self.ai_model.clone(),
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
    ) -> Result<Vec<ChatMessage>> {
        const CHUNK_START: &str = "=== DOCUMENT CHUNK START ===\n";
        const CHUNK_END: &str = "\n=== DOCUMENT CHUNK END ===";
        const SOURCE: &str = "\n[SOURCE] ";

        let retrieved = chunks
            .iter()
            .take(self.top_k)
            .filter(|p| p.similarity >= self.min_similarity)
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

        let mut history = Vec::with_capacity(1 + self.top_k);
        history.push(ChatMessage::system(base_prompt.to_string()));
        history.extend(retrieved);
        Ok(history)
    }
}
