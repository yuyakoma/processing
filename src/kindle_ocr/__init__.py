"""Kindle OCR to RAG pipeline package."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("kindle-rag")
except PackageNotFoundError:  # pragma: no cover - package not installed in dev mode
    __version__ = "0.0.0"

__all__ = ["__version__"]
