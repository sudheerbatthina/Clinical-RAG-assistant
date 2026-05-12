# ---------------------------------------------------------------------------
# Stage 1 — dependency layer (cached separately from source code)
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS deps

WORKDIR /install

# System libraries needed by unstructured / pdfplumber / chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
        libmagic1 \
        poppler-utils \
        tesseract-ocr \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/deps -r requirements.txt
RUN python -m spacy download en_core_web_sm


# ---------------------------------------------------------------------------
# Stage 2 — runtime image
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy system libs installed in stage 1
RUN apt-get update && apt-get install -y --no-install-recommends \
        libmagic1 \
        poppler-utils \
        tesseract-ocr \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages
COPY --from=deps /deps /usr/local

# Copy application source (everything except what's in .dockerignore)
COPY rag_assistant/ ./rag_assistant/
COPY api.py .

# Data and vector store directories are mounted at runtime via volumes;
# create empty placeholders so the app can start without them.
RUN mkdir -p data chroma_db .cache

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
