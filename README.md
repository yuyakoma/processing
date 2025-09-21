# Kindle OCR RAG Pipeline

End-to-end toolkit for capturing Kindle page images, building searchable PDFs with OCR, and exposing the content to Large Language Models via Retrieval-Augmented Generation (RAG) and Model Context Protocol (MCP).

## Quick Start

1. **Create the Conda environment**
   ```bash
   make env
   ```
   The helper script installs Tesseract, Poppler, and the Python dependencies.

2. **Open VSCode**
   The workspace config in `.vscode/settings.json` points the Python extension and terminal to the `kindle_ocr` environment. Adjust the interpreter path if your Conda installation lives somewhere other than `~/miniconda3`.

3. **Capture a book**
   ```bash
   kindle-pipeline capture --book-id my_book --config config/capture.example.yml
   ```
   Automates Kindle page export (currently ships with a stub you can extend to drive the Kindle UI).

4. **Run OCR + build PDFs**
   ```bash
   kindle-pipeline ocr --book-id my_book
   ```
   Processes screenshots into a text-searchable PDF and structured text files.

5. **Ingest into the RAG store**
   ```bash
   kindle-pipeline rag ingest --book-id my_book
   ```
   Embeds and stores the content in a Chroma vector database for downstream MCP tooling.

6. **Expose via MCP**
   ```bash
   kindle-pipeline mcp serve
   ```
   Launches a lightweight MCP-compatible server with tools for `ChatGPT` or `Claude`.

## Repository Layout

- `src/kindle_ocr/` — Python package with capture, OCR, and RAG modules.
- `scripts/` — Utility scripts (environment bootstrap, helpers).
- `config/` — YAML templates for capture automation & pipeline settings.
- `data/raw/` — Original page captures (`book_id/page_XXXX.png`).
- `data/interim/` — Intermediate artifacts (deskewed images, per-page OCR JSON).
- `data/processed/` — Final PDFs, text corpora, and vector store files.

## Next Steps

- Implement the UI automation hooks in `src/kindle_ocr/capture.py` for your OS/Kindle setup.
- Connect your preferred LLM providers by setting API keys in `.env` (see `config/defaults.example.env`).
- Expand tests in `tests/` to cover the pipeline pieces you customize.

