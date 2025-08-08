#[test]
fn test_read_pdf() {
    let pdf_path = "data/Candidates.pdf";
    let text = pdf_extract::extract_text(pdf_path);
    assert!(text.is_ok());

    let text = text.unwrap();
    assert!(!text.is_empty(), "Extracted text should not be empty");

    println!("{text}");
}
