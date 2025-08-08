use std::{collections::HashMap, future::Future};

use reqwest::{Client, IntoUrl, Method, Request, Url, redirect::Policy};

use crate::error::{RagError, RagResult};

pub trait Embedder {
    fn embed(&self, text: &str) -> impl Future<Output = RagResult<Vec<f32>>> + Send;
}

pub struct OllamaEmbedder {
    addr: Url,
    model: String,
    client: Client,
}

impl OllamaEmbedder {
    pub async fn new<A: IntoUrl, S: ToString>(addr: A, model: S) -> RagResult<Self> {
        let client = Client::builder().redirect(Policy::none()).build()?;

        Ok(Self {
            addr: addr.into_url()?,
            model: model.to_string(),
            client,
        })
    }
}

impl Embedder for OllamaEmbedder {
    async fn embed(&self, prompt: &str) -> RagResult<Vec<f32>> {
        let url = self
            .addr
            .join("/api/embeddings")
            .map_err(|e| RagError::Url(e.to_string()))?;

        let mut request = Request::new(Method::POST, url);
        request.headers_mut().insert(
            reqwest::header::CONTENT_TYPE,
            "application/x-www-form-urlencoded".parse().unwrap(),
        );

        let body = serde_json::to_string_pretty(&HashMap::from([
            ("model", self.model.as_str()),
            ("prompt", prompt),
        ]))?;
        request.body_mut().replace(body.into());

        let response = self.client.execute(request).await?.bytes().await?;

        serde_json::from_slice::<HashMap<String, Vec<f32>>>(&response)
            .map_err(|e| RagError::Serde(e.to_string()))
            .and_then(|json| {
                json.get("embedding")
                    .cloned()
                    .ok_or_else(|| RagError::Response("No embedding found in response".to_string()))
            })
    }
}
