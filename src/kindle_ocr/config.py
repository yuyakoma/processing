"""Configuration models and helpers for the Kindle OCR pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, validator


class CaptureAutomationConfig(BaseModel):
    driver: str = Field(
        ..., description="Identifier for the automation backend (e.g. macos_osa, pyautogui)."
    )
    kindle_window_title: str = Field(
        "Kindle", description="Window title used to focus the Kindle app."
    )
    hotkeys: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of actions to hotkeys (e.g. screenshot, next_page).",
    )


class CaptureConfig(BaseModel):
    output_dir: Path
    image_format: str = "png"
    dpi: int = 300
    include_margin: bool = False
    page_turn_delay: float = 1.0
    max_pages: Optional[int] = None
    automation: Optional[CaptureAutomationConfig] = None

    @validator("output_dir", pre=True)
    def _expand_output_dir(cls, value: Path | str) -> Path:
        return Path(value).expanduser().resolve()


class OCRConfig(BaseModel):
    language: str = "eng"
    preprocess: bool = True
    deskew: bool = True
    binarize: bool = True
    adaptive_threshold: bool = True
    vision_model: Optional[str] = None
    vision_model_temperature: float = 0.1
    vision_batch_size: int = 8


class PDFConfig(BaseModel):
    page_size: str = "auto"
    metadata: dict[str, str] = Field(default_factory=dict)


class RAGConfig(BaseModel):
    chunk_size: int = 800
    chunk_overlap: int = 120
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_store: Path = Field(default=Path("data/processed/default/chroma"))

    @validator("vector_store", pre=True)
    def _expand_vector_store(cls, value: Path | str) -> Path:
        return Path(value).expanduser().resolve()


class PipelineConfig(BaseModel):
    book_id: str
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    pdf: PDFConfig = Field(default_factory=PDFConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)

    @validator("raw_dir", "interim_dir", "processed_dir", pre=True)
    def _expand_dirs(cls, value: Path | str) -> Path:
        return Path(value).expanduser().resolve()


def load_pipeline_config(path: Path | str) -> PipelineConfig:
    """Load a pipeline configuration file.

    Args:
        path: Path to a YAML configuration file.

    Returns:
        Parsed ``PipelineConfig`` instance.
    """

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return PipelineConfig(**data)


def ensure_directories(cfg: PipelineConfig) -> None:
    """Ensure filesystem directories required by the pipeline exist."""

    for folder in [cfg.raw_dir, cfg.interim_dir, cfg.processed_dir, cfg.rag.vector_store]:
        folder.mkdir(parents=True, exist_ok=True)


class CaptureConfigFile(BaseModel):
    book: dict[str, str] | None = None
    capture: CaptureConfig


def load_capture_config(path: Path | str) -> CaptureConfig:
    data = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    capture_data = data.get('capture') if isinstance(data, dict) else data
    return CaptureConfig(**capture_data)
