"""macOS automation backend using AppleScript and screencapture."""

from __future__ import annotations

import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageChops, ImageStat
from rich.console import Console
from rich.prompt import Confirm, Prompt

from ..capture import CaptureBackend, plan_capture
from ..capture import list_expected_pages
from ..config import (
    AutoStopConfig,
    BookMetadata,
    CaptureAutomationConfig,
    CaptureConfig,
)
from ..kindle_library import KindleBook, discover_books, prompt_user_to_select
from ..title_detection import BookTitleDetector, TitleDetectionResult

logger = logging.getLogger(__name__)

console = Console()

SUPPORTED_FORMATS = {"png", "jpg", "jpeg", "tif", "tiff"}

MODIFIER_MAP = {
    "cmd": "command down",
    "command": "command down",
    "shift": "shift down",
    "ctrl": "control down",
    "control": "control down",
    "alt": "option down",
    "option": "option down",
}

KEYCODE_MAP = {
    "return": 36,
    "enter": 36,
    "space": 49,
    "tab": 48,
    "escape": 53,
    "esc": 53,
    "delete": 51,
    "backspace": 51,
    "forward_delete": 117,
    "home": 115,
    "end": 119,
    "pageup": 116,
    "pagedown": 121,
    "up": 126,
    "down": 125,
    "left": 123,
    "right": 124,
}

FUNCTION_KEYCODES = {
    1: 122,
    2: 120,
    3: 99,
    4: 118,
    5: 96,
    6: 97,
    7: 98,
    8: 100,
    9: 101,
    10: 109,
    11: 103,
    12: 111,
    13: 105,
    14: 107,
    15: 113,
    16: 106,
}


class MacOSAProxyCaptureBackend(CaptureBackend):
    """Automates Kindle page capture via AppleScript keystrokes."""

    def capture_book(self, cfg: CaptureConfig) -> None:
        automation_cfg = cfg.automation or CaptureAutomationConfig(driver="macos_osa")
        cfg.automation = automation_cfg

        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        selected_book = self._ensure_book_metadata(cfg, automation_cfg)
        self._ensure_book_defaults(cfg)
        self._ensure_output_directory(cfg, automation_cfg)

        plan_capture(cfg)
        console.print("[blue]Press Ctrl+C at any time to stop capturing.[/]")

        delay_seconds = max(float(getattr(automation_cfg, "focus_delay_seconds", 0.0) or 0.0), 0.0)
        if delay_seconds > 0:
            console.print(
                f"[cyan]Waiting {delay_seconds:.1f} seconds before starting. Bring the target window to the front now.[/]"
            )
            time.sleep(delay_seconds)

        resolved_proc_name = self._resolve_process_name(automation_cfg.kindle_window_title)
        self._focus_kindle(automation_cfg.kindle_window_title)
        self._maybe_open_book(selected_book)

        if automation_cfg.confirm_before_start and not automation_cfg.unattended:
            console.print(
                "\n[green]Align the Kindle window to the page you want to start from, then press Enter.[/]"
            )
            console.input("Ready to capture? Press Enter to begin… ")
        else:
            console.print(
                "\n[cyan]Auto-start enabled. Beginning capture in 2 seconds — press Ctrl+C to abort.[/]"
            )
            time.sleep(2.0)

        try:
            self._capture_loop(cfg, automation_cfg, resolved_proc_name)
        except KeyboardInterrupt:  # pragma: no cover - interactive flow
            console.print("\n[yellow]Capture interrupted by user.[/]")

    # ------------------------------------------------------------------
    # Discovery helpers
    def _ensure_book_metadata(
        self, cfg: CaptureConfig, automation_cfg: CaptureAutomationConfig
    ) -> KindleBook | None:
        current = cfg.book
        selection: KindleBook | None = None

        try:
            discovered = discover_books()
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.debug("Failed to scan Kindle library: %s", exc)
            discovered = []

        if discovered:
            if automation_cfg.auto_select_first_book or automation_cfg.unattended:
                selection = discovered[0]
                console.print(
                    f"[green]Auto-selected Kindle title:[/] {selection.display_label()}"
                )
            else:
                console.print("[green]Detected Kindle library on disk.[/]")
                selection = prompt_user_to_select(discovered, console)
        else:
            console.print(
                "[yellow]No Kindle library metadata detected. Continuing without automatic book selection.[/]"
            )

        if selection:
            book_id = (current.id if current and current.id else selection.asin) or self._slugify(selection.title)
            cfg.book = BookMetadata(id=book_id, title=selection.title, author=selection.author)
        elif not current or not current.title:
            if automation_cfg.unattended:
                console.print(
                    "[yellow]No title provided. Will attempt automatic detection from captured pages.[/]"
                )
                if not current:
                    cfg.book = BookMetadata(id=None, title=None, author=None)
            else:
                manual_title = Prompt.ask(
                    "Enter the book title to include in the capture plan (leave blank to skip)",
                    default="",
                    show_default=False,
                )
                if manual_title:
                    book_id = (current.id if current else None) or self._slugify(manual_title)
                    author = current.author if current else None
                    cfg.book = BookMetadata(id=book_id, title=manual_title, author=author)

        return selection

    def _ensure_output_directory(
        self, cfg: CaptureConfig, automation_cfg: CaptureAutomationConfig
    ) -> None:
        existing_files = [
            path
            for path in cfg.output_dir.iterdir()
            if path.is_file() and not path.name.startswith(".")
        ]
        if not existing_files:
            return

        console.print(
            f"[yellow]Output directory {cfg.output_dir} already contains {len(existing_files)} files.[/]"
        )

        if automation_cfg.unattended:
            console.print("[yellow]Unattended mode: clearing existing captures automatically.[/]")
            for path in existing_files:
                self._safe_remove(path)
            return

        if Confirm.ask("Delete existing files before capture?", default=False):
            for path in existing_files:
                self._safe_remove(path)
        else:
            raise RuntimeError("Capture aborted: output directory must be empty to avoid page drift.")

    # ------------------------------------------------------------------
    # Capture loop
    def _capture_loop(
        self,
        cfg: CaptureConfig,
        automation: CaptureAutomationConfig,
        process_name: str,
    ) -> None:
        image_format = cfg.image_format.lower()
        if image_format not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported image format '{cfg.image_format}'. Supported formats: {sorted(SUPPORTED_FORMATS)}"
            )

        next_key = (automation.hotkeys.get("next_page") if automation.hotkeys else None) or "right"
        kindle_process_name = process_name
        if kindle_process_name.lower() not in {"auto", "frontmost", "*"}:
            if not self._process_exists(kindle_process_name):
                console.print(
                    f"[yellow]Process '{kindle_process_name}' is not currently running. Using the frontmost window for capture.[/]"
                )
                kindle_process_name = "frontmost"

        console.print("\n[cyan]Starting automated capture. Press Ctrl+C to finish early.[/]")

        stop_policy: AutoStopConfig = getattr(automation, "stop_policy", AutoStopConfig())
        stop_enabled = stop_policy.enabled
        title_detector = None
        if automation.title_detection and automation.title_detection.enabled:
            title_detector = BookTitleDetector(automation.title_detection)

        duplicates: list[Path] = []
        reference_page: Path | None = None
        unique_pages = 0
        metadata_written = False

        for index, page_path in enumerate(list_expected_pages(cfg), start=1):
            console.print(f"Capturing page {index} -> {page_path.name}")

            window_id = self._front_window_id(kindle_process_name)
            if window_id is not None:
                self._screencapture_window(window_id, page_path, image_format)
            else:
                # Fallback: capture by rectangle using front window frame
                frame = self._front_window_frame(kindle_process_name)
                if frame is None:
                    self._report_missing_window(kindle_process_name)
                    break
                x, y, w, h = frame
                self._screencapture_rect(x, y, w, h, page_path, image_format)
            self._enforce_dpi(page_path, cfg.dpi)

            if reference_page is None:
                reference_page = page_path
                unique_pages += 1
                metadata_written = self._handle_title_detection(
                    title_detector, cfg, page_path, metadata_written
                )
            else:
                if stop_enabled:
                    diff_value = self._calculate_difference(reference_page, page_path)
                    logger.debug(
                        "Page difference (unique=%s, index=%s): %.3f",
                        unique_pages,
                        index,
                        diff_value,
                    )
                    if (
                        unique_pages >= stop_policy.warmup_unique_pages
                        and diff_value <= stop_policy.difference_threshold
                    ):
                        duplicates.append(page_path)
                        console.print(
                            f"[yellow]Page unchanged (Δ={diff_value:.2f}). Waiting for Kindle to advance…[/]"
                        )
                        if len(duplicates) >= stop_policy.repeat_limit:
                            console.print(
                                "[green]Page no longer changes after repeated turns. Stopping capture.[/]"
                            )
                            self._flush_duplicates(duplicates)
                            break

                        self._send_hotkey(next_key, kindle_process_name)
                        time.sleep(max(cfg.page_turn_delay, 0.25))
                        continue

                if duplicates:
                    console.print(
                        f"[yellow]Discarded {len(duplicates)} duplicate capture(s) after page change.[/]"
                    )
                    self._flush_duplicates(duplicates)

                reference_page = page_path
                unique_pages += 1
                metadata_written = self._handle_title_detection(
                    title_detector, cfg, page_path, metadata_written
                )

            if cfg.max_pages and index >= cfg.max_pages:
                break

            self._send_hotkey(next_key, kindle_process_name)
            time.sleep(max(cfg.page_turn_delay, 0.25))

    # ------------------------------------------------------------------
    # Title detection and housekeeping
    def _handle_title_detection(
        self,
        detector: BookTitleDetector | None,
        cfg: CaptureConfig,
        image_path: Path,
        metadata_written: bool,
    ) -> bool:
        if detector is None:
            return metadata_written

        try:
            result = detector.consider(image_path)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Title detection failed on %s: %s", image_path, exc)
            return metadata_written

        if not result:
            return metadata_written

        book = cfg.book or BookMetadata(id=None, title=None, author=None)
        updated = False

        if not book.title:
            book.title = result.title
            updated = True
        if not book.id:
            book.id = self._slugify(result.title)
            updated = True

        cfg.book = book

        if updated:
            console.print(f"[green]Detected book title:[/] {result.title} ({result.method})")

        if not metadata_written or updated:
            self._write_capture_metadata(cfg, result)
            metadata_written = True

        return metadata_written

    def _write_capture_metadata(self, cfg: CaptureConfig, detection: TitleDetectionResult) -> None:
        if not cfg.book:
            return

        payload = {
            "book": {
                "id": cfg.book.id,
                "title": cfg.book.title,
                "author": cfg.book.author,
            },
            "detection": {
                "method": detection.method,
                "confidence": detection.confidence,
                "raw": detection.raw,
            },
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }

        metadata_path = cfg.output_dir / "capture_metadata.json"
        try:
            metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:  # pragma: no cover - filesystem fallback
            logger.warning("Failed to write capture metadata to %s: %s", metadata_path, exc)

    def _flush_duplicates(self, duplicates: list[Path]) -> None:
        for path in duplicates:
            self._safe_remove(path)
        duplicates.clear()

    def _safe_remove(self, path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            return
        except Exception as exc:  # pragma: no cover - filesystem fallback
            logger.warning("Failed to remove %s: %s", path, exc)

    def _calculate_difference(self, first: Path, second: Path) -> float:
        try:
            with Image.open(first) as img_a, Image.open(second) as img_b:
                size = (512, 512)
                img_a = img_a.convert("L").resize(size)
                img_b = img_b.convert("L").resize(size)
                diff = ImageChops.difference(img_a, img_b)
                stat = ImageStat.Stat(diff)
                return float(stat.mean[0])
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.debug("Failed to compare %s and %s: %s", first, second, exc)
            return 255.0

    def _ensure_book_defaults(self, cfg: CaptureConfig) -> None:
        if cfg.book is None:
            cfg.book = BookMetadata(id=self._slugify(cfg.output_dir.name), title=None, author=None)
            return

        if not cfg.book.id:
            source = cfg.book.title or cfg.output_dir.name
            cfg.book.id = self._slugify(source)

    # ------------------------------------------------------------------
    # AppleScript helpers
    def _focus_kindle(self, identifier: str) -> None:
        # Resolve to a process name if possible; otherwise fall back to frontmost
        resolved = self._resolve_process_name(identifier)
        if resolved.lower() in {"auto", "frontmost", "*"}:
            # Leave current frontmost app as-is.
            return

        # Focus by app name (more tolerant). If this fails, continue without hard error.
        escaped_name = self._escape_applescript(resolved)
        script = f'''
try
tell application "{escaped_name}"
    activate
end tell
on error
    -- Some Catalyst apps do not expose a traditional AppleScript dictionary.
    -- Ignore and fall back to the accessibility-focused path below.
end try
delay 0.2
tell application "System Events"
    if exists process "{escaped_name}" then
        tell process "{escaped_name}"
            set frontmost to true
        end tell
    end if
end tell
'''
        try:
            self._run_osascript(script)
        except Exception as exc:  # pragma: no cover - fallback if focusing fails
            logger.warning("Unable to explicitly focus '%s' (%s). Continuing with current frontmost app.", resolved, exc)

    def _resolve_process_name(self, identifier: str) -> str:
        """Resolve a user-supplied identifier to a System Events process name."""

        ident = (identifier or "").strip()
        if not ident:
            return "frontmost"

        lowered = ident.lower()
        if lowered in {"auto", "frontmost", "*"}:
            return ident

        if ident.startswith("id:"):
            bundle_id_raw = ident[3:].strip()
            bundle_id = self._escape_applescript(bundle_id_raw)
            try:
                result = subprocess.run(
                    ["osascript", "-e", f'return name of application id "{bundle_id}"'],
                    text=True,
                    capture_output=True,
                    check=True,
                )
                name = (result.stdout or "").strip()
                if name:
                    return name
                logger.warning(
                    "Bundle ID '%s' did not resolve to an application name; falling back to frontmost window.",
                    bundle_id_raw,
                )
                return "frontmost"
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.warning(
                    "Failed to resolve application name for bundle id '%s': %s. Falling back to frontmost window.",
                    bundle_id_raw,
                    exc,
                )
                return "frontmost"

        processes = self._list_application_processes()
        if processes:
            for name in processes:
                if name.lower() == lowered:
                    return name

            partial_matches = [name for name in processes if lowered in name.lower()]
            if partial_matches:
                chosen = partial_matches[0]
                if chosen != ident:
                    console.print(
                        f"[yellow]Process '{ident}' not found. Using closest match '{chosen}' instead.[/]"
                    )
                return chosen

        if ident != "frontmost":
            console.print(
                "[yellow]Process '{ident}' not found among running apps. Using the frontmost window instead.[/]".format(
                    ident=ident
                )
            )
        return "frontmost"

    def _list_application_processes(self) -> list[str]:
        script = '''
tell application "System Events"
    try
        return name of every application process
    on error
        return ""
    end try
end tell
'''
        try:
            result = self._run_osascript(script)
        except Exception:
            return []

        output = (result.stdout or "").strip()
        if not output:
            return []

        # AppleScript returns a comma-separated list.
        return [item.strip() for item in output.split(",") if item.strip()]

    def _process_exists(self, process_name: str) -> bool:
        escaped = self._escape_applescript(process_name)
        script = f'''
tell application "System Events"
    try
        return (exists process "{escaped}")
    on error
        return false
    end try
end tell
'''
        try:
            result = self._run_osascript(script)
        except Exception:
            return False

        return (result.stdout or "").strip().lower() == "true"

    def _frontmost_process_name(self) -> str | None:
        script = '''
tell application "System Events"
    try
        set p to first process whose frontmost is true
        return name of p
    on error
        return ""
    end try
end tell
'''
        try:
            result = self._run_osascript(script)
        except Exception:
            return None

        name = (result.stdout or "").strip()
        return name or None

    def diagnose_window(self, identifier: str, output_path: Path | None = None) -> dict[str, object]:
        resolved = self._resolve_process_name(identifier)
        resolved_lower = resolved.lower()
        using_frontmost = resolved_lower in {"auto", "frontmost", "*"}

        process_running = True
        if not using_frontmost:
            process_running = self._process_exists(resolved)
            if process_running:
                self._focus_kindle(resolved)
                time.sleep(0.3)

        fallback_to_frontmost = not using_frontmost and not process_running
        target_identifier = "frontmost" if fallback_to_frontmost else resolved

        resolved_window_id = self._front_window_id(resolved)
        resolved_frame = self._front_window_frame(resolved)

        target_window_id = resolved_window_id if not fallback_to_frontmost else self._front_window_id("frontmost")
        target_frame = resolved_frame if not fallback_to_frontmost else self._front_window_frame("frontmost")

        capture_success = False
        capture_error: str | None = None
        capture_path_str: str | None = None
        path_to_use = output_path

        if path_to_use is not None:
            fmt = path_to_use.suffix.lower().lstrip(".")
            if not fmt:
                fmt = "png"
                path_to_use = path_to_use.with_suffix(".png")

            if fmt not in SUPPORTED_FORMATS:
                capture_error = f"unsupported format: {fmt}"
            elif target_window_id is None and target_frame is None:
                capture_error = "window_not_found"
            else:
                path_to_use.parent.mkdir(parents=True, exist_ok=True)
                try:
                    if target_window_id is not None:
                        self._screencapture_window(target_window_id, path_to_use, fmt)
                    elif target_frame is not None:
                        x, y, w, h = target_frame
                        self._screencapture_rect(x, y, w, h, path_to_use, fmt)
                    else:
                        capture_error = "window_not_found"
                except Exception as exc:
                    capture_error = str(exc)
                else:
                    capture_success = True
                    capture_path_str = str(path_to_use)

        return {
            "requested_identifier": identifier,
            "resolved_identifier": resolved,
            "using_frontmost": using_frontmost,
            "process_running": process_running,
            "fallback_to_frontmost": fallback_to_frontmost,
            "target_identifier": target_identifier,
            "frontmost_process": self._frontmost_process_name(),
            "resolved_window_id": resolved_window_id,
            "resolved_frame": resolved_frame,
            "target_window_id": target_window_id,
            "target_frame": target_frame,
            "capture_requested": output_path is not None,
            "capture_success": capture_success,
            "capture_path": capture_path_str,
            "capture_error": capture_error,
        }

    def _report_missing_window(self, process_name: str) -> None:
        console.print(
            "[red]Could not locate the target window. Both the accessibility window ID and frame were unavailable.[/]"
        )
        console.print(
            "[yellow]Make sure the Kindle window is frontmost and grant accessibility and screen-recording permissions to your terminal or IDE.[/]"
        )
        if process_name.lower() not in {"auto", "frontmost", "*"}:
            console.print(
                "[yellow]If the Kindle app runs under a different name, update 'automation.kindle_window_title' or set it to 'frontmost'.[/]"
            )
            candidates = [
                name for name in self._list_application_processes() if "kindle" in name.lower()
            ]
            if candidates:
                console.print(
                    "[yellow]Detected running apps containing 'Kindle': {}[/]".format(
                        ", ".join(sorted(set(candidates)))
                    )
                )

    def _frontmost_window_id(self) -> int | None:
        script = '''
tell application "System Events"
    try
        set p to first process whose frontmost is true
        return value of attribute "AXWindowNumber" of front window of p
    on error
        return ""
    end try
end tell
'''
        try:
            result = self._run_osascript(script)
        except Exception:
            return None
        output = result.stdout.strip()
        if not output:
            return None
        try:
            return int(output)
        except ValueError:
            return None

    def _maybe_open_book(self, selection: KindleBook | None) -> None:
        if not selection:
            return

        if selection.content_path and selection.content_path.exists():
            try:
                subprocess.run(["open", str(selection.content_path)], check=True)
                time.sleep(2.0)
                return
            except subprocess.CalledProcessError as exc:
                logger.debug("Failed to open book via content path: %s", exc)

        if selection.asin:
            try:
                subprocess.run(["open", f"kindle://book/{selection.asin}"], check=True)
                time.sleep(2.0)
                return
            except subprocess.CalledProcessError as exc:
                logger.debug("Failed to open book via kindle:// URL: %s", exc)

        console.print(
            "[yellow]Unable to automatically open the selected book. Please open it manually in Kindle.[/]"
        )

    def _front_window_id(self, process_name: str) -> int | None:
        if process_name.lower() in {"auto", "frontmost", "*"}:
            return self._frontmost_window_id()
        escaped_name = self._escape_applescript(process_name)
        script = f'''
tell application "System Events"
    if not (exists process "{escaped_name}") then
        return ""
    end if
    tell process "{escaped_name}"
        try
            return value of attribute "AXWindowNumber" of front window
        on error
            return ""
        end try
    end tell
end tell
'''
        try:
            result = self._run_osascript(script)
        except subprocess.CalledProcessError as exc:
            logger.error("Failed to query Kindle window id: %s", exc)
            return None

        output = result.stdout.strip()
        if not output:
            return None
        try:
            return int(output)
        except ValueError:
            logger.debug("Unexpected window id output: %s", output)
            return None

    def _send_hotkey(self, hotkey: str, process_name: str) -> None:
        action = self._render_hotkey(hotkey)
        if action is None:
            logger.debug("Skipping unknown hotkey: %s", hotkey)
            return

        if process_name.lower() in {"auto", "frontmost", "*"}:
            # Send to whatever app is frontmost
            script = f'tell application "System Events" to {action}\n'
        else:
            escaped_name = self._escape_applescript(process_name)
            script = f'''
tell application "System Events"
    if exists process "{escaped_name}" then
        tell process "{escaped_name}"
            set frontmost to true
            {action}
        end tell
    end if
end tell
'''
        try:
            self._run_osascript(script)
        except subprocess.CalledProcessError as exc:
            logger.debug("Failed to send hotkey '%s': %s", hotkey, exc)

    # ------------------------------------------------------------------
    # System command helpers
    def _screencapture_window(self, window_id: int, output_path: Path, fmt: str) -> None:
        cmd = ["screencapture", "-x", "-l", str(window_id)]
        format_flag = "jpg" if fmt in {"jpg", "jpeg"} else fmt
        if format_flag != "png":
            cmd.extend(["-t", format_flag])
        cmd.append(str(output_path))

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Failed to capture page screenshot: {exc}") from exc

    def _screencapture_rect(self, x: int, y: int, w: int, h: int, output_path: Path, fmt: str) -> None:
        # Coordinates are in global screen space. Negative values are allowed for displays
        # arranged left/above the main screen.
        cmd = ["screencapture", "-x", "-R", f"{x},{y},{w},{h}"]
        format_flag = "jpg" if fmt in {"jpg", "jpeg"} else fmt
        if format_flag != "png":
            cmd.extend(["-t", format_flag])
        cmd.append(str(output_path))

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Failed to capture rectangle screenshot: {exc}") from exc

    def _enforce_dpi(self, image_path: Path, dpi: int) -> None:
        if dpi <= 0:
            return
        try:
            with Image.open(image_path) as img:
                info = dict(img.info)
                fmt = img.format
                save_kwargs = {}
                if "icc_profile" in info:
                    save_kwargs["icc_profile"] = info["icc_profile"]
                img.save(image_path, format=fmt, dpi=(dpi, dpi), **save_kwargs)
        except Exception as exc:
            logger.debug("Failed to enforce DPI on %s: %s", image_path, exc)

    # ------------------------------------------------------------------
    # AppleScript plumbing
    def _run_osascript(self, script: str) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["osascript", "-"],
                input=script,
                text=True,
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            console.print(
                "[red]AppleScript 実行に失敗しました。" "'ターミナル' や使用中の IDE に対して、macOS のアクセス権 (アクセシビリティ/スクリーン収録) を付与してください。"
            )
            if stderr:
                console.print(f"[red]osascript stderr:[/] {stderr}")
            raise RuntimeError("AppleScript automation unavailable") from exc

    def _front_window_frame(self, process_name: str) -> tuple[int, int, int, int] | None:
        # Returns (x, y, w, h) of the front window for the given process, or the
        # system frontmost window if process_name is 'auto'/'frontmost'.
        if process_name.lower() in {"auto", "frontmost", "*"}:
            script = '''
tell application "System Events"
    try
        set p to first process whose frontmost is true
        set posXY to position of front window of p
        set sizeWH to size of front window of p
        set posX to item 1 of posXY as integer
        set posY to item 2 of posXY as integer
        set sizeW to item 1 of sizeWH as integer
        set sizeH to item 2 of sizeWH as integer
        return (posX as text) & "," & (posY as text) & "," & (sizeW as text) & "," & (sizeH as text)
    on error
        return ""
    end try
end tell
'''
        else:
            escaped = self._escape_applescript(process_name)
            script = f'''
tell application "System Events"
    if not (exists process "{escaped}") then
        return ""
    end if
    tell process "{escaped}"
        try
            set posXY to position of front window
            set sizeWH to size of front window
            set posX to item 1 of posXY as integer
            set posY to item 2 of posXY as integer
            set sizeW to item 1 of sizeWH as integer
            set sizeH to item 2 of sizeWH as integer
            return (posX as text) & "," & (posY as text) & "," & (sizeW as text) & "," & (sizeH as text)
        on error
            return ""
        end try
    end tell
end tell
'''
        try:
            result = self._run_osascript(script)
        except Exception:
            return None

        output = (result.stdout or "").strip()
        if not output:
            return None
        parts = output.split(",")
        if len(parts) != 4:
            return None
        try:
            x, y, w, h = map(int, parts)
            return x, y, w, h
        except ValueError:
            return None

    def _render_hotkey(self, hotkey: str) -> str | None:
        parts = [part.strip().lower() for part in hotkey.split("+") if part.strip()]
        if not parts:
            return None

        key_token = parts[-1]
        modifiers = [MODIFIER_MAP[part] for part in parts[:-1] if part in MODIFIER_MAP]

        if key_token in KEYCODE_MAP:
            command = f"key code {KEYCODE_MAP[key_token]}"
        elif key_token.startswith("f") and key_token[1:].isdigit():
            code = FUNCTION_KEYCODES.get(int(key_token[1:]))
            if code is None:
                return None
            command = f"key code {code}"
        elif len(key_token) == 1:
            command = f'keystroke "{self._escape_applescript(key_token)}"'
        elif key_token == "plus":
            command = 'keystroke "+"'
        else:
            command = f'keystroke "{self._escape_applescript(key_token)}"'

        if modifiers:
            command += f" using {{{', '.join(modifiers)}}}"
        return command

    @staticmethod
    def _slugify(value: str) -> str:
        cleaned = [ch.lower() if ch.isalnum() else "-" for ch in value.strip()]
        slug = "".join(cleaned)
        while "--" in slug:
            slug = slug.replace("--", "-")
        return slug.strip("-") or "book"

    @staticmethod
    def _escape_applescript(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')


__all__ = ["MacOSAProxyCaptureBackend"]
