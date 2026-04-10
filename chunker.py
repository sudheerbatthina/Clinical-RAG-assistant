from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from pdf_extractor import extract_text_from_pdf


def table_to_markdown(table: list[list[str]]) -> str:
    """Convert a pdfplumber table (list of rows) into a Markdown table string.

    pdfplumber returns tables as list[list[str|None]]. We treat None as empty,
    flatten internal newlines in cells, and produce a clean Markdown table
    that LLMs read fluently.
    """
    if not table or not table[0]:
        return ""
    
    def clean_cell(cell):
        if cell is None:
            return ""
        return cell.replace("\n", " ").strip()
    
    header = [clean_cell(c) for c in table[0]]
    body_rows = [[clean_cell(c) for c in row] for row in table[1:]]

    header_line = "| " + " | ".join(header) + " |"
    separator_line = "| " + " | ".join("---" for _ in header) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in body_rows]

    return "\n".join([header_line, separator_line] + body_lines)


def chunk_pages(pages: list[dict], source_name: str) -> list[dict]:
    """Split extracted pages into retrieval-ready chunks.

    Prose text is split with a recursive character splitter. Tables are
    converted to Markdown and kept whole as a single chunk each.

    Args:
        pages: Output of extract_text_from_pdf — list of page dicts.
        source_name: The source filename, attached to every chunk for citation.

    Returns:
        A list of chunk dicts. Each chunk has keys:
        - chunk_id: stable, unique identifier
        - source: source filename
        - page_number: 1-indexed page number
        - chunk_type: "text" or "table"
        - content: the actual string the LLM will see
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for page in pages:
        page_number = page["page_number"]

        # Prose text chunks
        text_pieces = splitter.split_text(page["text"])
        for i, piece in enumerate(text_pieces):
            chunks.append({
                "chunk_id": f"{source_name}_p{page_number}_t{i}",
                "source": source_name,
                "page_number": page_number,
                "chunk_type": "text",
                "content": piece,
            })

        
        # Table chunks
        for table_idx, table in enumerate(page["tables"]):
            markdown = table_to_markdown(table)
            if not markdown:
                continue
            chunks.append({
                "chunk_id": f"{source_name}_p{page_number}_tbl{table_idx}",
                "source": source_name,
                "page_number": page_number,
                "chunk_type": "table",
                "content": markdown,
            })

    return chunks   


if __name__ == "__main__":
    pdf_path = Path("data/cms_hipaa.pdf")
    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_pages(pages, source_name=pdf_path.name)

    print(f"Produced {len(chunks)} chunks from {len(pages)} pages")
    print()

    text_chunks = [c for c in chunks if c["chunk_type"] == "text"]
    table_chunks = [c for c in chunks if c["chunk_type"] == "table"]
    print(f"  text chunks:  {len(text_chunks)}")
    print(f"  table chunks: {len(table_chunks)}")
    print()

    print("--- First text chunk ---")
    print(text_chunks[0]["chunk_id"])
    print(text_chunks[0]["content"][:400])
    print()

    if table_chunks:
        print("--- First table chunk ---")
        print(table_chunks[0]["chunk_id"])
        print(table_chunks[0]["content"][:600])