use anyhow::{Context, Result};
use qdrant_client::{
    Qdrant,
    qdrant::{
        CreateCollectionBuilder, Distance, PointStruct, ScalarQuantizationBuilder, ScoredPoint,
        SearchPointsBuilder, UpsertPointsBuilder, VectorParamsBuilder,
    },
};

use crate::calculate_hash;

#[derive(Debug)]
pub struct RetrievedChunk {
    pub content: String,
    pub source: String,
    pub similarity: f32,
}

pub trait VectorDB {
    async fn add_vector(&self, payload: String, source: String, vector: Vec<f32>) -> Result<()>;
    async fn find(&self, vector: Vec<f32>) -> Result<Vec<RetrievedChunk>>;
}

pub struct QdrantDB {
    client: Qdrant,
    collection: String,
}

impl QdrantDB {
    pub async fn new(url: &str, collection: &str, vector_size: u64) -> Result<Self> {
        let client = Qdrant::from_url(url).build()?;

        if !client.collection_exists(collection).await? {
            client
                .create_collection(
                    CreateCollectionBuilder::new(collection).vectors_config(
                        VectorParamsBuilder::new(vector_size, Distance::Euclid)
                            .quantization_config(ScalarQuantizationBuilder::default()),
                    ),
                )
                .await?;
        }

        Ok(QdrantDB {
            client,
            collection: collection.to_string(),
        })
    }

    pub async fn connect(url: &str, collection: &str) -> Result<Self> {
        let client = Qdrant::from_url(url).build()?;
        Ok(QdrantDB {
            client,
            collection: collection.to_string(),
        })
    }
}

impl VectorDB for QdrantDB {
    async fn add_vector(&self, payload: String, source: String, vector: Vec<f32>) -> Result<()> {
        let id = calculate_hash(&payload.to_string());

        let point = PointStruct::new(
            id,
            vector,
            [("value", payload.into()), ("source", source.into())],
        );
        self.client
            .upsert_points(UpsertPointsBuilder::new(
                self.collection.clone(),
                vec![point],
            ))
            .await?;

        Ok(())
    }

    async fn find(&self, vector: Vec<f32>) -> Result<Vec<RetrievedChunk>> {
        let collection = self.collection.clone();
        let search_request = SearchPointsBuilder::new(collection, vector, 5).with_payload(true);

        let chunks = self
            .client
            .search_points(search_request)
            .await?
            .result
            .iter()
            .map(|x| {
                let content = extract(x, "value")?;
                let source = extract(x, "source")?;
                let similarity = x.score;

                Ok(RetrievedChunk {
                    content,
                    source,
                    similarity,
                })
            })
            .collect::<Result<_>>()?;
        Ok(chunks)
    }
}

fn extract(x: &ScoredPoint, key: &str) -> Result<String> {
    x.payload
        .get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .with_context(|| format!("Failed to extract {} from payload", key))
}
