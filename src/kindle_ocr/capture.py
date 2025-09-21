"""Capture Kindle pages into image files.

This module exposes a simple interface that can be tailored to different
platforms or automation backends. The default implementation only scaffolds
hooks and records the intended capture plan.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from rich.console import Console
from rich.table import Table

from .config import CaptureConfig

logger = logging.getLogger(__name__)

console = Console()


def list_expected_pages(cfg: CaptureConfig) -> Iterable[Path]:
    """Return the expected filenames for each captured page."""

    output_dir = cfg.output_dir
    max_pages = cfg.max_pages or 9999
    width = len(str(max_pages))
    for index in range(1, max_pages + 1):
        yield output_dir / f"page_{index:0{width}d}.{cfg.image_format}"


def plan_capture(cfg: CaptureConfig) -> None:
    """Display a capture plan so the user can verify settings."""

    table = Table(title="Kindle Capture Plan")
    table.add_column("Setting")
    table.add_column("Value")

    table.add_row("Output directory", str(cfg.output_dir))
    table.add_row("Image format", cfg.image_format)
    table.add_row("DPI", str(cfg.dpi))
    table.add_row("Max pages", str(cfg.max_pages or "auto"))
    table.add_row("Page turn delay", f"{cfg.page_turn_delay:.1f}s")
    table.add_row(
        "Automation driver",
        cfg.automation.driver if cfg.automation else "manual",
    )

    console.print(table)


class CaptureBackend:
    """Abstract capture backend interface."""

    def capture_book(self, cfg: CaptureConfig) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class ManualCaptureBackend(CaptureBackend):
    """Manual capture backend which only prepares folders and instructions."""

    def capture_book(self, cfg: CaptureConfig) -> None:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        plan_capture(cfg)
        console.print(
            "[yellow]Manual mode:[/] Use your preferred tool to capture each page. "
            "Name files following the template shown in the plan."
        )


BACKENDS: dict[str, type[CaptureBackend]] = {
    "manual": ManualCaptureBackend,
}


try:
    from .drivers.macos_osa import MacOSAProxyCaptureBackend  # type: ignore

    BACKENDS["macos_osa"] = MacOSAProxyCaptureBackend
except Exception:  # pragma: no cover - optional dependency
    logger.debug("macOS OSA backend unavailable.")


def get_backend(name: str) -> CaptureBackend:
    backend_cls = BACKENDS.get(name)
    if backend_cls is None:
        raise ValueError(f"Unsupported capture backend: {name}")
    return backend_cls()


def run_capture(cfg: CaptureConfig) -> None:
    backend_name = cfg.automation.driver if cfg.automation else "manual"
    backend = get_backend(backend_name)
    backend.capture_book(cfg)
