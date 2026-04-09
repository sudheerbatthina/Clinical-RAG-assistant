from pathlib import Path

from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """Extract text from a PDF file, page by page.

    Args:
        pdf_path: Path to the PDF file to read.

    Returns:
        A list of dictionaries, one per page, each with keys
        'page_number' (1-indexed) and 'text' (the extracted text). """
    
    reader = PdfReader(pdf_path)
    extracted_pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        extracted_pages.append({
            "page_number": page_number,
            "text": page.extract_text()
        })
    return extracted_pages


if __name__ == "__main__":
    pdf_path = Path("data/cms_hipaa.pdf")
    pages = extract_text_from_pdf(pdf_path)

    print(f"Extracted {len(pages)} pages from {pdf_path.name}")
    print()
    print("--- First page preview ---")
    print(pages[0]["text"][:500])