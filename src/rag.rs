use std::{fs, path::Path};

use text_splitter::{Characters, TextSplitter};

use crate::{Embedder, RagResult, VectorDB, calculate_hash};

const HASH_DB: &str = "hash.db";

pub struct Rag<E: Embedder, D: VectorDB> {
    embedder: E,
    vector_db: D,
    splitter: TextSplitter<Characters>,
}

impl<E: Embedder, D: VectorDB> Rag<E, D> {
    pub fn new(embedder: E, vector_db: D, text_splitter: usize) -> Self {
        let splitter = TextSplitter::new(text_splitter);

        Self::init_hash_db();

        Rag {
            embedder,
            vector_db,
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
                    Some(ext) if ext == "txt" => fs::read_to_string(path)?,
                    _ => continue,
                };

                self.embed_text(&text).await?;
            }
        }

        Ok(())
    }

    pub async fn embed_text(&mut self, text: &str) -> RagResult<()> {
        let hash = calculate_hash(&text.to_string());
        if Self::has_hash(hash) {
            return Ok(());
        }

        let chunks = self.splitter.chunks(text);
        for chunk in chunks {
            let vector = self.embedder.embed(&chunk).await?;
            self.vector_db.add_vector(chunk, vector).await?;
        }

        Self::add_hash(hash);
        Ok(())
    }

    pub async fn search_embedding(
        &mut self,
        text: &str,
    ) -> RagResult<Vec<(std::string::String, f32)>> {
        let vector = self.embedder.embed(text).await?;
        let x = self.vector_db.query_vector(vector).await?;
        Ok(x)
    }

    fn init_hash_db() {
        if !Path::new(HASH_DB).exists() {
            fs::File::create(HASH_DB).unwrap();
        }
    }

    fn has_hash(hash: u64) -> bool {
        let data = fs::read(HASH_DB).unwrap();
        data.chunks_exact(8)
            .into_iter()
            .map(|x| u64::from_le_bytes(x.try_into().unwrap()))
            .find(|&x| x == hash)
            .is_some()
    }

    fn add_hash(hash: u64) {
        let mut data = fs::read(HASH_DB).unwrap_or_default();
        data.extend_from_slice(&hash.to_le_bytes());
        fs::write(HASH_DB, data).unwrap();
    }
}
