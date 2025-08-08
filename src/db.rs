use std::{
    future::Future,
    hash::{DefaultHasher, Hash, Hasher},
};

use qdrant_client::{
    Qdrant,
    qdrant::{
        CreateCollectionBuilder, Distance, PointStruct, ScalarQuantizationBuilder,
        SearchPointsBuilder, UpsertPointsBuilder, VectorParamsBuilder,
    },
};

use crate::RagResult;

pub trait VectorDB {
    fn add_vector(
        &mut self,
        payload: &str,
        vector: Vec<f32>,
    ) -> impl Future<Output = RagResult<()>> + Send;

    fn query_vector(
        &self,
        vector: Vec<f32>,
    ) -> impl Future<Output = RagResult<Vec<(String, f32)>>> + Send;
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

    pub async fn connect(url: &str, collection: &str) -> Self {
        let client = Qdrant::from_url(url).build().unwrap();
        QdrantDB {
            client,
            collection: collection.to_string(),
        }
    }
}

impl VectorDB for QdrantDB {
    async fn add_vector(&mut self, payload: &str, vector: Vec<f32>) -> RagResult<()> {
        let mut hasher = DefaultHasher::new();
        payload.hash(&mut hasher);
        let id = hasher.finish();

        let point = PointStruct::new(id, vector, [("value", payload.into())]);
        self.client
            .upsert_points(UpsertPointsBuilder::new(
                self.collection.clone(),
                vec![point],
            ))
            .await?;

        Ok(())
    }

    async fn query_vector(&self, vector: Vec<f32>) -> RagResult<Vec<(String, f32)>> {
        let search_request =
            SearchPointsBuilder::new(self.collection.clone(), vector, 10).with_payload(true);

        Ok(self
            .client
            .search_points(search_request)
            .await?
            .result
            .iter()
            .map(|x| (x.payload["value"].as_str().unwrap().to_string(), x.score))
            .collect())
    }
}
