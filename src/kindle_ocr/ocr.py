"""OCR processing utilities for Kindle captures."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytesseract
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import OCRConfig

logger = logging.getLogger(__name__)


@dataclass
class OCRPageResult:
    page_path: Path
    text: str
    confidence: float
    engine: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "page_path": str(self.page_path),
                "text": self.text,
                "confidence": self.confidence,
                "engine": self.engine,
            },
            ensure_ascii=False,
        )


class OCREngine:
    def __init__(self, cfg: OCRConfig):
        self.cfg = cfg

    def run(self, image_path: Path, output_json: Optional[Path] = None) -> OCRPageResult:
        img = self._load_image(image_path)

        logger.debug("Running OCR on %s", image_path)
        text, confidence = self._run_tesseract(img)
        engine_used = "tesseract"

        if not text.strip() and self.cfg.vision_model:
            logger.info("Tesseract returned empty text for %s; invoking %s", image_path, self.cfg.vision_model)
            text = self._run_vision_model(img)
            confidence = 0.8 if text else 0.0
            engine_used = self.cfg.vision_model

        result = OCRPageResult(page_path=image_path, text=text.strip(), confidence=confidence, engine=engine_used)

        if output_json:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(result.to_json(), encoding="utf-8")

        return result

    def _load_image(self, image_path: Path) -> np.ndarray:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        logger.debug("Loaded image %s with shape %s", image_path, img.shape)
        return img

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self.cfg.deskew:
            gray = self._deskew(gray)

        if self.cfg.binarize:
            if self.cfg.adaptive_threshold:
                gray = cv2.adaptiveThreshold(
                    gray,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    35,
                    11,
                )
            else:
                _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return cv2.fastNlMeansDenoising(gray)

    def _deskew(self, gray: np.ndarray) -> np.ndarray:
        coords = np.column_stack(np.where(gray > 0))
        if coords.size == 0:
            return gray
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = gray.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        logger.debug("Deskewed image by %.2f degrees", angle)
        return rotated

    def _run_tesseract(self, img: np.ndarray) -> tuple[str, float]:
        processed = self._preprocess(img) if self.cfg.preprocess else img
        config = r"--oem 3 --psm 6"
        data = pytesseract.image_to_data(processed, lang=self.cfg.language, config=config, output_type=pytesseract.Output.DICT)

        text = " ".join(word for word in data["text"] if word.strip())
        confidences = [float(conf) for conf in data.get("conf", []) if conf not in {"-1", "-1.0"}]
        confidence = float(np.mean(confidences)) / 100 if confidences else 0.0

        return text, confidence

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _run_vision_model(self, img: np.ndarray) -> str:
        if not self.cfg.vision_model:
            return ""

        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError("httpx is required for vision model OCR") from exc

        logger.warning("Vision model support is stubbed. Provide your own integration.")

        # Placeholder for custom implementation: return empty string to indicate failure.
        return ""


def process_book(cfg: OCRConfig, image_dir: Path, output_dir: Path) -> list[OCRPageResult]:
    """Run OCR across a directory of images."""
    engine = OCREngine(cfg)
    results: list[OCRPageResult] = []

    images = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}]
    )
    if not images:
        logger.warning("No images found in %s", image_dir)
        return results

    for image_path in images:
        output_json = output_dir / f"{image_path.stem}.json"
        result = engine.run(image_path, output_json=output_json)
        results.append(result)

    return results
