use std::{
    ffi::OsStr,
    fs::{self},
    path::{Path, PathBuf},
};

use text_splitter::{Characters, TextSplitter};

use crate::{Embedder, HashStorage, RagResult, VectorDB, calculate_hash, db::RetrievedChunk};

pub struct RagIndex<E: Embedder, D: VectorDB, S: HashStorage> {
    embedder: E,
    vector_db: D,
    storage: S,
    splitter: TextSplitter<Characters>,
}

impl<E: Embedder, D: VectorDB, S: HashStorage> RagIndex<E, D, S> {
    pub fn new(embedder: E, vector_db: D, storage: S, text_splitter: usize) -> Self {
        let splitter = TextSplitter::new(text_splitter);

        RagIndex {
            embedder,
            vector_db,
            storage,
            splitter,
        }
    }

    pub async fn embed_directory<P: AsRef<Path>>(&mut self, path: P) -> RagResult<()> {
        let entries = std::fs::read_dir(path)?;

        for entry in entries {
            let path = entry?.path();
            self.process_file(path).await?;
        }

        Ok(())
    }

    async fn process_file(&mut self, path: PathBuf) -> RagResult<bool> {
        if path.is_file() {
            let text = match path.extension() {
                Some(ext) if ext == "pdf" => pdf_extract::extract_text(&path)?,
                Some(ext) if ext == "txt" => fs::read_to_string(&path)?,
                _ => return Ok(false),
            };

            let filename = path
                .file_name()
                .unwrap_or(OsStr::new("unknown"))
                .to_string_lossy()
                .to_string();

            self.embed_text(text, filename).await?;
        }

        Ok(true)
    }

    pub async fn embed_text(&mut self, text: String, source: String) -> RagResult<()> {
        let hash = calculate_hash(&text);
        if self.storage.contains(hash)? {
            return Ok(());
        }

        let chunks = self.splitter.chunks(&text);
        for chunk in chunks {
            let vector = self.embedder.embed(chunk).await?;
            self.vector_db
                .add_vector(chunk.to_string(), source.clone(), vector)
                .await?;
        }

        self.storage.insert(hash)
    }

    pub async fn search_embedding(&mut self, prompt: &str) -> RagResult<Vec<RetrievedChunk>> {
        let vector = self.embedder.embed(prompt).await?;
        let x = self.vector_db.query_vector(vector).await?;
        Ok(x)
    }
}
