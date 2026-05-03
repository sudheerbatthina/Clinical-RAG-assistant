from pathlib import Path

import pdfplumber
from unstructured.partition.pdf import partition_pdf


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """Extract text using unstructured, tables using pdfplumber."""
    # Prose extraction via unstructured (better column handling)
    elements = partition_pdf(filename=str(pdf_path), strategy="hi_res")

    pages = {}
    for el in elements:
        page_num = el.metadata.page_number or 1
        if page_num not in pages:
            pages[page_num] = {"page_number": page_num, "text": "", "tables": []}
        try:
            text = el.text or ""
        except AttributeError:
            text = ""
        if text:
            pages[page_num]["text"] += text + "\n"

    # Table extraction via pdfplumber (reliable table detection)
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            if i not in pages:
                pages[i] = {"page_number": i, "text": "", "tables": []}
            pages[i]["tables"] = page.extract_tables()

    return [pages[k] for k in sorted(pages.keys())]


if __name__ == "__main__":
    pdf_path = Path("data/cms_hipaa.pdf")
    result = extract_text_from_pdf(pdf_path)

    print(f"Extracted {len(result)} pages")
    print(f"Page 1 text preview:\n{result[0]['text'][:300]}\n")
    print(f"Page 12 tables: {len(result[11]['tables'])}")
    if result[11]["tables"]:
        print(f"First table rows: {len(result[11]['tables'][0])}")