use std::{ffi::OsStr, fs, path::Path};

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
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                let text = match path.extension() {
                    Some(ext) if ext == "pdf" => pdf_extract::extract_text(&path)?,
                    Some(ext) if ext == "txt" => fs::read_to_string(&path)?,
                    _ => continue,
                };

                let filename = path
                    .file_name()
                    .unwrap_or(OsStr::new("unknown"))
                    .to_string_lossy()
                    .to_string();

                self.embed_text(text, filename).await?;
            }
        }

        Ok(())
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

    pub async fn search_embedding(&mut self, text: &str) -> RagResult<Vec<RetrievedChunk>> {
        let vector = self.embedder.embed(text).await?;
        let x = self.vector_db.query_vector(vector).await?;
        Ok(x)
    }
}
