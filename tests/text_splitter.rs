use std::sync::LazyLock;

use text_splitter::TextSplitter;

const PDF_DOCUMENT: &[u8] = include_bytes!("../data/Candidates.txt");
static PDF_TEXT: LazyLock<String> = LazyLock::new(|| {
    let text = pdf_extract::extract_text_from_mem(PDF_DOCUMENT);
    assert!(text.is_ok());
    text.unwrap()
});

#[test]
fn test_text_splitter() {
    let max_characters = 600;
    let splitter = TextSplitter::new(max_characters);

    let chunks: Vec<String> = splitter.chunks(&PDF_TEXT).map(|x| x.to_string()).collect();
    assert!(!chunks.is_empty(), "Chunks should not be empty");
}
