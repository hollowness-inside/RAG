use anyhow::Result;
use rag::RagChain;

#[tokio::main]
async fn main() -> Result<()> {
    let mut chain = RagChain::builder()
        .set_embed_model("mxbai-embed-large".to_string())
        .set_ai_model("qwen3:latest".to_string())
        .build_default()
        .await?;

    chain.embed_directory("./data/").await?;

    let response = chain.ask("Who is Elara?").await?;
    println!("Response: {:?}", response.content);
    Ok(())
}
