# NOTE: pdfplumber mixes multi-column headers on page 1.
# Acceptable for single-column body text; revisit if processing
# multi-column documents in production.from pathlib import Path

from pypdf import PdfReader

import pdfplumber

def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """Extract text from a PDF file, page by page.

    Args:
        pdf_path: Path to the PDF file to read.

    Returns:
        A list of dictionaries, one per page, each with keys:
        - 'page number' (1-indexed)
        - 'text' (the extracted text as string)
        - 'tables' (a list of tables, where each table is a list of rows, and each row is a list of cell values)
    """
        
    #reader = PdfReader(pdf_path)
    extracted_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            extracted_pages.append({
                "page_number": page_number,
                "text": page.extract_text() or "",
                "tables": page.extract_tables(),
        })
    return extracted_pages


if __name__ == "__main__":
    pdf_path = Path("data/cms_hipaa.pdf")
    pages = extract_text_from_pdf(pdf_path)

    print(f"Extracted {len(pages)} pages from {pdf_path.name}")
    print()

    print("--- Page 1 text preview ---")
    print(pages[0]["text"][:500])
    print()

    print("--- Page 12 text preview ---")
    print(pages[11]["text"][:500])
    print()

    print(f"--- Page 12 has {len(pages[11]['tables'])} table(s) detected ---")
    if pages[11]["tables"]:
        first_table = pages[11]["tables"][0]
        print(f"First table has {len(first_table)} rows")
        print("First 3 rows:")
        for row in first_table[:3]:
            print(row)