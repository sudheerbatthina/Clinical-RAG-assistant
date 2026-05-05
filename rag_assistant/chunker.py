from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_SIZE, CHUNK_OVERLAP, SECTION_MAX_CHARS


def table_to_markdown(table: list[list[str]]) -> str:
    """Convert a pdfplumber table (list of rows) into a Markdown table string."""
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


def _split_at_paragraphs(text: str, max_size: int) -> list[str]:
    """Split text at paragraph boundaries, keeping parts under max_size."""
    parts = []
    current = ""

    for para in text.split("\n\n"):
        if not current:
            current = para
        elif len(current) + 2 + len(para) <= max_size:
            current += "\n\n" + para
        else:
            parts.append(current)
            current = para
    if current:
        parts.append(current)

    # Second pass: if any part still exceeds max_size, split at newlines
    final = []
    for part in parts:
        if len(part) <= max_size:
            final.append(part)
        else:
            curr = ""
            for line in part.split("\n"):
                if not curr:
                    curr = line
                elif len(curr) + 1 + len(line) <= max_size:
                    curr += "\n" + line
                else:
                    final.append(curr)
                    curr = line
            if curr:
                final.append(curr)

    return [p for p in final if p.strip()]


_SECTION_BOUNDARY_TYPES = {"Title", "Header"}


def chunk_document(
    elements: list,
    tables_by_page: dict,
    source_name: str,
) -> list[dict]:
    """Section-aware chunking: group elements by Title/Header boundaries.

    Each section becomes one chunk. Sections larger than SECTION_MAX_CHARS
    are split at paragraph boundaries. Tables are kept whole as single chunks.

    Args:
        elements: Raw unstructured elements (each has .category, .text, .metadata).
        tables_by_page: {page_num: [table_list]} from pdfplumber.
        source_name: Filename attached to every chunk for citation.

    Returns:
        List of chunk dicts with keys: chunk_id, source, page_number,
        chunk_type, content.
    """
    # --- Group elements into sections ---
    sections: list[dict] = []
    current: dict = {"text": "", "page": None}

    for el in elements:
        page_num = el.metadata.page_number or 1
        try:
            text = el.text or ""
        except AttributeError:
            text = ""

        if el.category in _SECTION_BOUNDARY_TYPES:
            if current["text"].strip():
                sections.append(current)
            current = {
                "text": (text + "\n") if text else "",
                "page": page_num,
            }
        else:
            if text:
                if current["page"] is None:
                    current["page"] = page_num
                current["text"] += text + "\n"

    if current["text"].strip():
        sections.append(current)

    # Fall back to empty section list guard
    if not sections:
        return []

    # --- Convert sections to chunks ---
    chunks: list[dict] = []
    for sec_idx, section in enumerate(sections):
        page = section["page"] or 1
        text = section["text"].strip()

        if len(text) <= SECTION_MAX_CHARS:
            chunks.append({
                "chunk_id": f"{source_name}_s{sec_idx}",
                "source": source_name,
                "page_number": page,
                "chunk_type": "text",
                "content": text,
            })
        else:
            parts = _split_at_paragraphs(text, SECTION_MAX_CHARS)
            for part_idx, part in enumerate(parts):
                chunks.append({
                    "chunk_id": f"{source_name}_s{sec_idx}_p{part_idx}",
                    "source": source_name,
                    "page_number": page,
                    "chunk_type": "text",
                    "content": part,
                })

    # --- Table chunks (unchanged behaviour) ---
    for page_num, tables in tables_by_page.items():
        for tbl_idx, table in enumerate(tables):
            markdown = table_to_markdown(table)
            if not markdown:
                continue
            chunks.append({
                "chunk_id": f"{source_name}_p{page_num}_tbl{tbl_idx}",
                "source": source_name,
                "page_number": page_num,
                "chunk_type": "table",
                "content": markdown,
            })

    return chunks


def chunk_pages(pages: list[dict], source_name: str) -> list[dict]:
    """Legacy text-split chunker kept for backward compatibility.

    Uses RecursiveCharacterTextSplitter on pre-aggregated page text.
    Prefer chunk_document() for new ingestion pipelines.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for page in pages:
        page_number = page["page_number"]

        for i, piece in enumerate(splitter.split_text(page["text"])):
            chunks.append({
                "chunk_id": f"{source_name}_p{page_number}_t{i}",
                "source": source_name,
                "page_number": page_number,
                "chunk_type": "text",
                "content": piece,
            })

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
