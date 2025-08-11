use std::{
    ffi::OsStr,
    fs::{self},
    path::{Path, PathBuf},
};

use anyhow::Result;
use text_splitter::{Characters, TextSplitter};

use crate::{
    db::{RetrievedChunk, VectorDB},
    embedder::Embedder,
};

pub struct RagIndex<E: Embedder, D: VectorDB> {
    embedder: E,
    vector_db: D,
    splitter: TextSplitter<Characters>,
}

impl<E: Embedder, D: VectorDB> RagIndex<E, D> {
    pub fn new(embedder: E, vector_db: D, text_splitter: usize) -> Self {
        let splitter = TextSplitter::new(text_splitter);

        RagIndex {
            embedder,
            vector_db,
            splitter,
        }
    }

    pub async fn embed_directory<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let entries = std::fs::read_dir(path)?;

        for entry in entries {
            let path = entry?.path();
            self.process_file(path).await?;
        }

        Ok(())
    }

    pub async fn embed_text(&mut self, text: String, source: String) -> Result<()> {
        let chunks = self.splitter.chunks(&text);
        for chunk in chunks {
            let vector = self.embedder.embed(chunk).await?;
            self.vector_db
                .add_vector(chunk.to_string(), source.clone(), vector)
                .await?;
        }
        Ok(())
    }

    async fn process_file(&mut self, path: PathBuf) -> Result<bool> {
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

    pub async fn search(&mut self, prompt: &str) -> Result<Vec<RetrievedChunk>> {
        let vector = self.embedder.embed(prompt).await?;
        let chunks = self.vector_db.find(vector).await?;
        Ok(chunks)
    }
}
