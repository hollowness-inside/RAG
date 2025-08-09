# RAG

A lightweight, modular Rust implementation of a Retrieval-Augmented Generation (RAG) pipeline. The design emphasizes composability: every major behavior (embedding, retrieval, orchestration, agent logic) is driven by traits so you can plug in your own implementations without forking core code.

## Why this project
- Trait-first architecture: swap components at compile time.
- Minimal surface area: start with a working pipeline fast.
- Extensible: add new embedding backends, retrievers, ranking stages, or agent behaviors.
- Async-ready (Tokio) for concurrent ingestion and querying.

## High-level flow
1. Configure (RagConfig) the chain (RagChain).
2. Ingest content (e.g. embed_directory).
3. Ask a question (ask) → retrieval + augmentation + generation.

## Quick start
```
gh repo clone hollowness-inside/RAG
cd RAG
cargo run
```

Example (from src/main.rs):
```rust
let mut chain = RagChain::new().await?;

chain.embed_directory("./data/").await?;

let response = chain.ask("Who is Elara?").await?;
println!("{}", response.content);
```

Response (truncated)

```
Elara is a gifted young alchemist whose silver hair and twilight-colored eyes mark her as a figure of both beauty and tragedy [Source <source name 1>]. Once the most promising apprentice of her mentor, Kael, her life was irrevocably altered by a catastrophic explosion that killed both her and Kael, leading the Guild to exile her to the Rib-Cages District, a shadowy underbelly of the city [Source <source name 1>]. There, she sustains herself by crafting illicit Lumina potions, which provide survival for the city's marginalized population [Source <source name 1>].

Her quest for redemption and truth drives her to unravel the cryptic clues in Kael's journal, a task she undertakes with the aid of Silas, a historian and linguist who provides her with the journal [Source <source name 5>]. Through this pursuit, she discovers that Kael's final lesson involves deciphering metaphors like the "sky-serpent's tooth," which she interprets as a rare crystalline formation in Atheria's Cranial Palaces [Source <source name 3>]. Now, as a guardian of the Sunstone, she and Silas work to forge a future free from the Guild's exploitation, embodying Kael's belief that "true alchemy is not about transforming metals, but about transforming the world" [Source <source name 4>].
```

## Contributing
1. Open an issue describing the new trait or extension point.
2. Provide a reference implementation + doc example.
3. Add concise benchmarks if performance-relevant.

## License
MIT license

## Inspiration
Built to stay minimal while enabling experimentation with modern RAG patterns.

Simply learning AIs.

Happy hacking—extend by implementing a trait, not by rewriting the core.