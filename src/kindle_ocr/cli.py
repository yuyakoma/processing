"""Command line interface for the Kindle OCR pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .config import CaptureConfig, PipelineConfig, ensure_directories, load_capture_config, load_pipeline_config
from .pipeline import run_capture_stage, run_ocr_stage, run_pdf_stage, run_rag_stage, run_pipeline
from .rag import RAGStore

console = Console()
app = typer.Typer(help="Kindle OCR + RAG toolkit")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


@app.command()
def capture(
    config: Path = typer.Argument(..., help="Path to capture configuration YAML"),
) -> None:
    """Run the capture workflow (scaffold)."""

    capture_cfg: CaptureConfig = load_capture_config(config)
    run_capture_stage(capture_cfg)


@app.command()
def ocr(
    config: Path = typer.Argument(..., help="Path to pipeline configuration YAML"),
) -> None:
    """Run OCR for the specified book configuration."""

    pipeline_cfg: PipelineConfig = load_pipeline_config(config)
    ensure_directories(pipeline_cfg)
    run_ocr_stage(pipeline_cfg)


@app.command()
def pdf(config: Path = typer.Argument(..., help="Path to pipeline configuration YAML")) -> None:
    """Rebuild searchable PDF for the specified book."""

    pipeline_cfg: PipelineConfig = load_pipeline_config(config)
    ensure_directories(pipeline_cfg)
    ocr_results = run_ocr_stage(pipeline_cfg)
    if not ocr_results:
        console.print("[red]No OCR results found. Run the OCR step first.")
        raise typer.Exit(code=1)
    pdf_path = run_pdf_stage(pipeline_cfg, ocr_results)
    console.print(f"[green]Searchable PDF saved to {pdf_path}")


@app.command()
def rag(
    config: Path = typer.Argument(..., help="Path to pipeline configuration YAML"),
    query: Optional[str] = typer.Option(None, help="Optional query to test retrieval after ingestion"),
    top_k: int = typer.Option(5, help="Number of matches to return when querying"),
) -> None:
    """Ingest OCR output into the vector store and optionally run a sample query."""

    pipeline_cfg: PipelineConfig = load_pipeline_config(config)
    ensure_directories(pipeline_cfg)
    ocr_results = run_ocr_stage(pipeline_cfg)
    run_rag_stage(pipeline_cfg, ocr_results)

    if query:
        store = RAGStore(pipeline_cfg)
        results = store.query(query, top_k=top_k)
        if not results:
            console.print("[yellow]No results found.")
        for idx, result in enumerate(results, start=1):
            console.print(f"[{idx}] dist={result['distance']:.3f} page={result['metadata']['page']}")
            console.print(result["document"][:400] + "\n")


@app.command()
def full(config: Path = typer.Argument(..., help="Path to pipeline configuration YAML")) -> None:
    """Run the full pipeline (OCR → PDF → RAG)."""

    pipeline_cfg: PipelineConfig = load_pipeline_config(config)
    pdf_path = run_pipeline(pipeline_cfg)
    console.print(f"[green]Pipeline complete. PDF saved to {pdf_path}")


@app.command()
def mcp(config: Path = typer.Argument(..., help="Path to pipeline configuration YAML")) -> None:
    """Stub command to start the MCP server."""

    console.print(
        "[yellow]MCP server integration is not yet implemented. "
        "Use the RAG store via the 'rag' command for now."
    )


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
