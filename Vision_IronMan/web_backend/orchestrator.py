from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .store import ScanStore


PROJECT_DIR = Path(__file__).resolve().parents[1]
PIPELINE_PATH = PROJECT_DIR / "qwen_module_fai_pipeline.py"
POPPLER_BIN = Path(
    "/Users/macsuper/.cache/codex-runtimes/codex-primary-runtime/"
    "dependencies/bin/override"
)


def executable(name: str) -> str:
    bundled = POPPLER_BIN / name
    return str(bundled) if bundled.is_file() else name


def validate_pdf(path: Path) -> None:
    if path.stat().st_size < 5 or path.read_bytes()[:5] != b"%PDF-":
        raise ValueError("上传文件不是有效 PDF")


def render_pdf(pdf_path: Path, pages_dir: Path) -> list[Path]:
    pages_dir.mkdir(parents=True, exist_ok=True)
    prefix = pages_dir / "page"
    command = [
        executable("pdftoppm"),
        "-png",
        "-r",
        os.environ.get("CLAW_VIEW_PDF_DPI", "180"),
        str(pdf_path),
        str(prefix),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "PDF 页面渲染失败")
    return sorted(
        pages_dir.glob("page-*.png"),
        key=lambda path: int(path.stem.rsplit("-", 1)[-1]),
    )


def build_pipeline_command(
    page_path: Path,
    output_dir: Path,
    events_file: Path,
    page_index: int,
    page_total: int,
) -> list[str]:
    command = [
        sys.executable,
        str(PIPELINE_PATH),
        str(page_path),
        "--output",
        str(output_dir),
        "--events-file",
        str(events_file),
        "--page-index",
        str(page_index),
        "--page-total",
        str(page_total),
    ]
    option_map = {
        "CLAW_VIEW_ENDPOINT": "--endpoint",
        "CLAW_VIEW_MODEL": "--model",
        "CLAW_VIEW_RECOVERY_ENDPOINT": "--recovery-endpoint",
        "CLAW_VIEW_RECOVERY_MODEL": "--recovery-model",
        "CLAW_VIEW_API_KEY": "--api-key",
    }
    for environment_key, option in option_map.items():
        value = os.environ.get(environment_key)
        if value:
            command.extend([option, value])
    return command


def consume_new_events(
    events_file: Path,
    offset: int,
    store: ScanStore,
    run_id: str,
) -> int:
    if not events_file.exists():
        return offset
    with events_file.open("r", encoding="utf-8") as handle:
        handle.seek(offset)
        while line := handle.readline():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            event_type = event.get("type")
            if event_type == "stage1_complete":
                store.save_stage1_progress(run_id, event)
            elif event_type == "stage2_module_complete":
                store.save_stage2_progress(run_id, event)
            elif event_type == "refined_crop_ready":
                refined_path = Path(event["output_dir"]) / event["refined_crop_path"]
                if not refined_path.is_file():
                    continue
                event["refined_crop_path_absolute"] = str(refined_path.resolve())
                store.upsert_refined_record(run_id, event)
        return handle.tell()


def process_pdf_run(store: ScanStore, run_id: str) -> None:
    run = store.get_run(run_id)
    if not run:
        return
    run_dir = Path(run["run_dir"])
    try:
        store.update_run(run_id, status="rendering", error=None)
        pdf_path = Path(run["pdf_path"])
        validate_pdf(pdf_path)
        pages = render_pdf(pdf_path, run_dir / "pages")
        if not pages:
            raise RuntimeError("PDF 中没有可处理的页面")
        store.update_run(run_id, status="processing", total_pages=len(pages))

        for page_index, page_path in enumerate(pages, start=1):
            store.update_run(run_id, current_page=page_index, status="processing")
            output_dir = run_dir / "pipeline" / f"page_{page_index:04d}"
            events_file = output_dir / "ui_events.jsonl"
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path = output_dir / "pipeline.log"
            command = build_pipeline_command(
                page_path, output_dir, events_file, page_index, len(pages)
            )
            with log_path.open("w", encoding="utf-8") as log_handle:
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_DIR,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                offset = 0
                while process.poll() is None:
                    offset = consume_new_events(events_file, offset, store, run_id)
                    time.sleep(0.5)
                consume_new_events(events_file, offset, store, run_id)
            # A partial pipeline result still yields valid refined rows. Continue pages.

        store.update_run(run_id, status="completed", current_page=len(pages))
    except Exception as exc:
        store.update_run(run_id, status="failed", error=f"{type(exc).__name__}: {exc}")
