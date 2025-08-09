use std::{
    fs::File,
    path::{Path, PathBuf},
};

use crate::RagResult;

pub trait Storage {
    fn contains(&self, hash: u64) -> RagResult<bool>;
    fn insert(&mut self, hash: u64) -> RagResult<()>;
}

pub struct FileStorage(PathBuf);

impl FileStorage {
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        if path.as_ref().exists() {
            File::create(&path).unwrap();
        }

        Self(path.as_ref().into())
    }
}

impl Storage for FileStorage {
    fn contains(&self, hash: u64) -> RagResult<bool> {
        Ok(std::fs::read(&self.0)?
            .chunks_exact(8)
            .map(|x| u64::from_le_bytes(x.try_into().unwrap()))
            .any(|x| x == hash))
    }

    fn insert(&mut self, hash: u64) -> RagResult<()> {
        let mut data = std::fs::read(&self.0).unwrap_or_default();
        data.extend_from_slice(&hash.to_le_bytes());
        std::fs::write(&self.0, data).unwrap();
        Ok(())
    }
}
