"""macOS Automation backend using AppleScript via osascript."""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

from rich.console import Console

from ..config import CaptureConfig
from ..capture import CaptureBackend, plan_capture

logger = logging.getLogger(__name__)

OSA_TEMPLATE = """
set kindleApp to "Kindle"
if application kindleApp is running then
    tell application kindleApp to activate
end if
""".strip()

console = Console()


class MacOSAProxyCaptureBackend(CaptureBackend):
    """Automates Kindle page capture using AppleScript and screencapture."""

    def capture_book(self, cfg: CaptureConfig) -> None:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        plan_capture(cfg)
        logger.info("Starting macOS OSA capture")

        script_path = cfg.output_dir / "capture.applescript"
        script_path.write_text(OSA_TEMPLATE, encoding="utf-8")

        self._focus_kindle(script_path)
        console.print(
            "[green]macOS automation scaffold ready.[/] Extend 'capture.applescript' to "
            "send page turns and trigger your preferred screenshot workflow."
        )

    def _focus_kindle(self, script_path: Path) -> None:
        try:
            subprocess.run(["osascript", str(script_path)], check=True)
            time.sleep(0.5)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - macOS specific
            logger.error("Failed to execute AppleScript: %s", exc)


__all__ = ["MacOSAProxyCaptureBackend"]
