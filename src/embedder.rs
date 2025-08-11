use std::collections::HashMap;

use anyhow::{Context, Result};
use reqwest::{Client, IntoUrl, Method, Request, Url, redirect::Policy};

pub trait Embedder {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
}

pub struct OllamaEmbedder {
    addr: Url,
    model: String,
    client: Client,
}

impl OllamaEmbedder {
    pub async fn new<A: IntoUrl, S: ToString>(addr: A, model: S) -> Result<Self> {
        let client = Client::builder().redirect(Policy::none()).build()?;

        Ok(Self {
            addr: addr.into_url()?,
            model: model.to_string(),
            client,
        })
    }
}

impl Embedder for OllamaEmbedder {
    async fn embed(&self, prompt: &str) -> Result<Vec<f32>> {
        let url = self.addr.join("/api/embeddings")?;

        let mut request = Request::new(Method::POST, url);
        request
            .headers_mut()
            .insert(reqwest::header::CONTENT_TYPE, "application/json".parse()?);

        let body = serde_json::to_string(&HashMap::from([
            ("model", self.model.as_str()),
            ("prompt", prompt),
        ]))?;
        request.body_mut().replace(body.into());

        let response = self.client.execute(request).await?.bytes().await?;

        serde_json::from_slice::<HashMap<String, Vec<f32>>>(&response)?
            .get("embedding")
            .cloned()
            .with_context(|| {
                format!(
                    "Failed to extract embedding from response: {}",
                    String::from_utf8_lossy(&response)
                )
            })
    }
}
