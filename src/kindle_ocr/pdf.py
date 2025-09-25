"""PDF utilities for Kindle OCR pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import img2pdf
from pypdf import PdfReader, PdfWriter
from pypdf.annotations import FreeText

from .config import PDFConfig

logger = logging.getLogger(__name__)


def images_to_pdf(image_paths: Iterable[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as pdf_file:
        pdf_bytes = img2pdf.convert([str(path) for path in image_paths])
        pdf_file.write(pdf_bytes)
    logger.info("Wrote image PDF to %s", output_path)


def merge_pdf_with_text_layer(image_pdf: Path, ocr_texts: Iterable[str], output_path: Path, cfg: PDFConfig) -> None:
    """Combine the image-only PDF with hidden text layer placeholders."""

    reader = PdfReader(str(image_pdf))
    writer = PdfWriter()
    metadata = {f"/{key.title()}": value for key, value in cfg.metadata.items()}

    texts = list(ocr_texts)
    if len(texts) != len(reader.pages):
        logger.warning(
            "Number of OCR texts (%s) does not match PDF pages (%s)",
            len(texts),
            len(reader.pages),
        )

    for page, text in zip(reader.pages, texts):
        writer.add_page(page)
        if text:
            annotation = FreeText(
                text=text,
                rect=(0, 0, 0, 0),
                font="Helvetica",
                font_size="12pt",
                font_color="000000",
                border_color=None,
                background_color=None,
            )
            writer.add_annotation(page_number=len(writer.pages) - 1, annotation=annotation)

    if metadata:
        writer.add_metadata(metadata)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        writer.write(handle)
    logger.info("Created searchable PDF at %s", output_path)
