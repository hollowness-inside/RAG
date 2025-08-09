use rag::{RagChain, RagConfig, RagResult};

#[tokio::main]
async fn main() -> RagResult<()> {
    let cfg = RagConfig::default();
    let mut chain = RagChain::with_config(cfg).await?;

    chain.embed_directory("./data/").await?;

    let response = chain.ask("What is CPP?").await?;
    println!("Response: {:?}", response.content);
    Ok(())
}
