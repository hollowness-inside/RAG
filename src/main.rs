use rag::{RagChain, RagResult};

#[tokio::main]
async fn main() -> RagResult<()> {
    let mut chain = RagChain::new().await?;

    chain.embed_directory("./data/").await?;

    let response = chain.ask("What is CPP?").await?;
    println!("Response: {:?}", response.content);
    Ok(())
}
