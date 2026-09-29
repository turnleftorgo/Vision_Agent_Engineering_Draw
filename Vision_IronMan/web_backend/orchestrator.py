from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

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


def run_log(run_id: str, message: str, *, component: str = "BACKEND") -> None:
    """Write a human-readable business log line to the launcher terminal."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}][RUN {run_id}][{component}] {message}", flush=True)


def stream_pipeline_output(
    process: subprocess.Popen[str],
    log_handle: object,
    run_id: str,
    page_index: int,
) -> None:
    """Tee pipeline stdout to its durable log and the launcher terminal."""
    stdout = process.stdout
    if stdout is None:
        return
    for line in stdout:
        text = line.rstrip("\n")
        if not text:
            continue
        log_handle.write(line)  # type: ignore[union-attr]
        log_handle.flush()  # type: ignore[union-attr]
        run_log(run_id, f"page {page_index}: {text}", component="PIPELINE")


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
    logger: Callable[[str], None] | None = None,
    on_record_ready: Callable[[str, str], None] | None = None,
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
                if logger:
                    logger(f"Stage 1 complete: {event.get('module_count', 0)} module(s)")
            elif event_type == "stage2_module_started":
                store.update_run(
                    run_id,
                    current_module=int(event.get("module_index", 0)),
                )
                if logger:
                    logger(
                        "Stage 2 module %s/%s started"
                        % (event.get("module_index", "?"), event.get("module_total", "?"))
                    )
            elif event_type == "stage2_module_complete":
                store.save_stage2_progress(run_id, event)
                if int(event.get("page_index") or 0) == 1:
                    metadata_result = store.consolidate_page1_metadata(
                        run_id, int(event.get("module_total") or 0)
                    )
                    if logger and metadata_result.get("status") == "complete":
                        logger(
                            "page 1 metadata consolidated: filled=%s conflicts=%s"
                            % (
                                metadata_result.get("filled", []),
                                metadata_result.get("conflicts", []),
                            )
                        )
                store.update_run(
                    run_id,
                    current_module=int(event.get("module_index", 0)),
                )
                if logger:
                    logger(
                        "Stage 2 module %s complete: %s FAI cluster(s)"
                        % (
                            event.get("module_index", "?"),
                            event.get("valid_cluster_count", 0),
                        )
                    )
            elif event_type in {"refinement_started", "refinement_round", "refinement_finished"}:
                store.update_refinement(run_id, event)
            elif event_type == "refined_crop_ready":
                refined_path = Path(event["output_dir"]) / event["refined_crop_path"]
                if not refined_path.is_file():
                    continue
                event["refined_crop_path_absolute"] = str(refined_path.resolve())
                record_id = store.upsert_refined_record(run_id, event)
                if on_record_ready:
                    on_record_ready(run_id, record_id)
                if logger:
                    logger(
                        "refined ready: FAI %s (%s)"
                        % (event.get("fai_number", "—"), event.get("candidate_key", "?"))
                    )
        return handle.tell()


def process_pdf_run(
    store: ScanStore,
    run_id: str,
    on_record_ready: Callable[[str, str], None] | None = None,
) -> None:
    run = store.get_run(run_id)
    if not run:
        return
    run_dir = Path(run["run_dir"])
    try:
        log = lambda message: run_log(run_id, message)
        run_log(run_id, f"run started: {run['pdf_name']}")
        store.update_run(run_id, status="rendering", error=None)
        pdf_path = Path(run["pdf_path"])
        validate_pdf(pdf_path)
        run_log(run_id, f"rendering PDF: {pdf_path.name}")
        render_started = time.perf_counter()
        pages = render_pdf(pdf_path, run_dir / "pages")
        if not pages:
            raise RuntimeError("PDF 中没有可处理的页面")
        run_log(
            run_id,
            f"PDF rendered: {len(pages)} page(s) in {time.perf_counter() - render_started:.1f}s",
        )
        store.update_run(run_id, status="processing", total_pages=len(pages))

        for page_index, page_path in enumerate(pages, start=1):
            store.update_run(
                run_id,
                current_page=page_index,
                current_module=0,
                status="processing",
            )
            run_log(run_id, f"page {page_index}/{len(pages)} started")
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
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                output_thread = threading.Thread(
                    target=stream_pipeline_output,
                    args=(process, log_handle, run_id, page_index),
                    name=f"run-{run_id}-page-{page_index}-log",
                    daemon=True,
                )
                output_thread.start()
                offset = 0
                while process.poll() is None:
                    offset = consume_new_events(
                        events_file, offset, store, run_id, log, on_record_ready
                    )
                    time.sleep(0.5)
                output_thread.join(timeout=5)
                consume_new_events(
                    events_file, offset, store, run_id, log, on_record_ready
                )
                run_log(run_id, f"page {page_index}/{len(pages)} pipeline exited with code {process.returncode}")
            # A partial pipeline result still yields valid refined rows. Continue pages.

        store.update_run(run_id, status="completed", current_page=len(pages))
        run_log(run_id, "run completed")
    except Exception as exc:
        store.update_run(run_id, status="failed", error=f"{type(exc).__name__}: {exc}")
        run_log(run_id, f"run failed: {type(exc).__name__}: {exc}")
