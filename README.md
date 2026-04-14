# Clinical RAG Assistant

A Retrieval-Augmented Generation (RAG) pipeline for querying healthcare policy documents. It extracts text and tables from PDFs, embeds them with OpenAI, stores them in ChromaDB, and answers natural-language questions with cited sources.

## Pipeline

```
PDF → pdf_extractor → chunker → embedder → vector_store → retriever → generator
```

| Step | File | What it does |
|---|---|---|
| Extract | `pdf_extractor.py` | Pulls text and tables from each page using `pdfplumber` |
| Chunk | `chunker.py` | Splits prose with a recursive splitter; converts tables to Markdown blocks |
| Embed | `embedder.py` | Batches all chunks through OpenAI `text-embedding-3-small` |
| Store | `vector_store.py` | Persists embeddings + metadata to a local ChromaDB collection |
| Retrieve | `retriever.py` | Embeds a query and returns the top-k nearest chunks |
| Generate | `generator.py` | Sends retrieved context to `gpt-4o-mini` and returns a cited answer |

## Setup

### Prerequisites

- Python 3.11+
- An OpenAI API key

### Install

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

## Usage

### 1. Add your PDF

Place your PDF in the `data/` directory:

```
data/your_document.pdf
```

### 2. Run the ingestion pipeline

Each script can be run standalone. Run them in order:

```bash
# Extract → chunk → embed → store
python vector_store.py
```

Or step through individually to inspect intermediate output:

```bash
python pdf_extractor.py   # Preview extracted pages
python chunker.py         # Preview chunks
python embedder.py        # Preview embeddings
python vector_store.py    # Build the ChromaDB collection
```

> Update the `pdf_path` variable at the bottom of each script to point to your file.

### 3. Ask a question

```bash
python generator.py
```

Edit the `question` variable in `generator.py` to ask your own questions. The answer will include page citations from the source document.

**Example output:**

```
Q: What information must a CSR verify before releasing beneficiary information?

A: A CSR must verify ... (page 4)

Sources used:
  - page 4 (cms_hipaa.pdf_p4_t0)
  - page 7 (cms_hipaa.pdf_p7_t1)
```

## Project Structure

```
rag-assistant/
├── data/               # PDF documents (git-ignored)
├── chroma_db/          # Persistent vector store (git-ignored)
├── pdf_extractor.py    # PDF → pages (text + tables)
├── chunker.py          # Pages → retrieval-ready chunks
├── embedder.py         # Chunks → OpenAI embeddings
├── vector_store.py     # Embeddings → ChromaDB
├── retriever.py        # Query → top-k chunks
├── generator.py        # Question → cited answer
└── requirements.txt
```

## Dependencies

Key packages (see `requirements.txt` for pinned versions):

- `pdfplumber` — layout-aware PDF extraction with table detection
- `langchain-text-splitters` — recursive prose chunking
- `openai` — embeddings (`text-embedding-3-small`) and chat (`gpt-4o-mini`)
- `chromadb` — local persistent vector store
- `python-dotenv` — API key management
