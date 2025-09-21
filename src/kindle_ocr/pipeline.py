"""End-to-end orchestration helpers for the Kindle pipeline."""

from __future__ import annotations

import logging
from itertools import tee
from pathlib import Path
from typing import Iterable, Iterator

from .capture import run_capture
from .config import CaptureConfig, PipelineConfig, ensure_directories
from .ocr import OCRPageResult, OCREngine
from .pdf import images_to_pdf, merge_pdf_with_text_layer
from .rag import RAGStore

logger = logging.getLogger(__name__)


def run_capture_stage(capture_cfg: CaptureConfig) -> None:
    logger.info("Running capture stage")
    run_capture(capture_cfg)


def run_ocr_stage(cfg: PipelineConfig) -> list[OCRPageResult]:
    logger.info("Running OCR stage for %s", cfg.book_id)
    image_dir = cfg.raw_dir
    output_dir = cfg.interim_dir / "ocr_json"
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = OCREngine(cfg.ocr)
    results: list[OCRPageResult] = []

    images = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    )
    if not images:
        logger.warning("No images found for OCR stage in %s", image_dir)
        return results

    for image_path in images:
        json_path = output_dir / f"{image_path.stem}.json"
        result = engine.run(image_path, output_json=json_path)
        results.append(result)

    jsonl_path = cfg.interim_dir / "ocr_results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(result.to_json() + "\n")

    logger.info("Wrote OCR JSONL to %s", jsonl_path)
    return results


def run_pdf_stage(cfg: PipelineConfig, ocr_results: Iterable[OCRPageResult]) -> Path:
    logger.info("Running PDF assembly stage for %s", cfg.book_id)
    image_paths = sorted(
        p for p in cfg.raw_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    )
    pdf_dir = cfg.processed_dir
    pdf_dir.mkdir(parents=True, exist_ok=True)

    image_pdf = pdf_dir / f"{cfg.book_id}_images.pdf"
    images_to_pdf(image_paths, image_pdf)

    text_pdf = pdf_dir / f"{cfg.book_id}_searchable.pdf"
    texts = [result.text for result in ocr_results]
    merge_pdf_with_text_layer(image_pdf, texts, text_pdf, cfg.pdf)
    return text_pdf


def chunk_text(text: str, chunk_size: int, overlap: int) -> Iterator[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    step = chunk_size - overlap
    start = 0
    while start < len(text):
        yield text[start : start + chunk_size]
        start += step


def build_rag_chunks(ocr_results: Iterable[OCRPageResult], cfg: PipelineConfig) -> Iterator[tuple[str, dict[str, str]]]:
    for index, result in enumerate(ocr_results, start=1):
        for chunk_index, chunk in enumerate(
            chunk_text(result.text, cfg.rag.chunk_size, cfg.rag.chunk_overlap), start=1
        ):
            metadata = {
                "book_id": cfg.book_id,
                "page": str(index),
                "chunk": str(chunk_index),
            }
            yield chunk, metadata


def run_rag_stage(cfg: PipelineConfig, ocr_results: Iterable[OCRPageResult]) -> None:
    logger.info("Running RAG ingestion for %s", cfg.book_id)

    rag_store = RAGStore(cfg)
    ocr_results, ocr_results_copy = tee(ocr_results)
    rag_store.ingest(build_rag_chunks(ocr_results, cfg))

    preview_path = cfg.processed_dir / "ocr_text_preview.txt"
    with preview_path.open("w", encoding="utf-8") as handle:
        for result in ocr_results_copy:
            handle.write(f"# Page {result.page_path.stem}\n")
            handle.write(result.text + "\n\n")
    logger.info("Saved OCR text preview to %s", preview_path)


def run_pipeline(cfg: PipelineConfig) -> Path:
    ensure_directories(cfg)
    ocr_results = run_ocr_stage(cfg)
    pdf_path = run_pdf_stage(cfg, ocr_results)
    run_rag_stage(cfg, ocr_results)
    return pdf_path
