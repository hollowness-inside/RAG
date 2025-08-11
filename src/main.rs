use anyhow::Result;
use rag::{RagChain, RagConfig};

#[tokio::main]
async fn main() -> Result<()> {
    let config = RagConfig {
        embed_model: "mxbai-embed-large".to_string(),
        ..Default::default()
    };

    let mut chain = RagChain::with_config(config).await?;

    chain.embed_directory("./data/").await?;

    let response = chain.ask("Who is Elara?").await?;
    println!("Response: {:?}", response.content);
    Ok(())
}
