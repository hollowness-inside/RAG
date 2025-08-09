use qdrant_client::{
    Qdrant,
    qdrant::{
        CreateCollectionBuilder, Distance, PointStruct, ScalarQuantizationBuilder, ScoredPoint,
        SearchPointsBuilder, UpsertPointsBuilder, VectorParamsBuilder,
    },
};

use crate::{RagError, RagResult, calculate_hash};

pub struct RetrievedChunk {
    pub content: String,
    pub source: String,
    pub similarity: f32,
}

pub trait VectorDB {
    async fn add_vector(&self, payload: String, source: String, vector: Vec<f32>) -> RagResult<()>;
    async fn query_vector(&self, vector: Vec<f32>) -> RagResult<Vec<RetrievedChunk>>;
}

pub struct QdrantDB {
    client: Qdrant,
    collection: String,
}

impl QdrantDB {
    pub async fn new(url: &str, collection: &str, vector_size: u64) -> RagResult<Self> {
        let client = Qdrant::from_url(url).build()?;

        if !client.collection_exists(collection).await? {
            client
                .create_collection(
                    CreateCollectionBuilder::new(collection).vectors_config(
                        VectorParamsBuilder::new(vector_size, Distance::Cosine)
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

    pub async fn connect(url: &str, collection: &str) -> RagResult<Self> {
        let client = Qdrant::from_url(url).build()?;
        Ok(QdrantDB {
            client,
            collection: collection.to_string(),
        })
    }
}

impl VectorDB for QdrantDB {
    async fn add_vector(&self, payload: String, source: String, vector: Vec<f32>) -> RagResult<()> {
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

    async fn query_vector(&self, vector: Vec<f32>) -> RagResult<Vec<RetrievedChunk>> {
        let collection = self.collection.clone();
        let search_request = SearchPointsBuilder::new(collection, vector, 10).with_payload(true);

        let chunks = self
            .client
            .search_points(search_request)
            .await?
            .result
            .iter()
            .map(|x| {
                Ok(RetrievedChunk {
                    content: extract(x, "value")?,
                    source: extract(x, "source")?,
                    similarity: x.score,
                })
            })
            .collect::<RagResult<Vec<RetrievedChunk>>>()?;
        Ok(chunks)
    }
}

fn extract(x: &ScoredPoint, key: &str) -> RagResult<String> {
    x.payload
        .get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or(RagError::Payload(key.to_string()))
}
