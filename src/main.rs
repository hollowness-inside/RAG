use rag::{RagChain, RagConfig, RagResult};

#[tokio::main]
async fn main() -> RagResult<()> {
    let config = RagConfig {
        embed_model: "mxbai-embed-large".to_string(),
        ..Default::default()
    };

    let mut chain = RagChain::with_config(config).await?;

    chain.embed_directory("./data/").await?;

    let response = chain.ask("CPP").await?;
    println!("Response: {:?}", response.content);
    Ok(())
}
