#!/usr/bin/env python3
"""Compare serial and overlapping Stage 2/Stage 3 calls to one model server.

Run this while no other pipeline or client is using the target server. The
script replays saved inputs without changing the selected run or its database.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sqlite3
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any, Callable

from PIL import Image

import qwen_module_fai_pipeline as pipeline
from web_backend.model_settings import get_enabled_proxy_url


ROOT = Path(__file__).resolve().parent
HTTP_STATUS_RE = re.compile(r"(?:Error code:|HTTP)\s*(\d{3})")


def schema_parameter_enabled() -> bool:
    """Inspect executable Python, since run_config currently has a static label."""
    tree = ast.parse(Path(pipeline.__file__).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "call_recovery_agent":
            return any(
                keyword.arg == "response_format"
                for call in ast.walk(node)
                if isinstance(call, ast.Call)
                for keyword in call.keywords
            )
    raise RuntimeError("call_recovery_agent was not found")


def pick_inputs(page_dir: Path, candidate_key: str | None) -> dict[str, Any]:
    config_path = page_dir / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    candidates = (
        [page_dir / "stage3_recovery" / candidate_key]
        if candidate_key
        else sorted((page_dir / "stage3_recovery").glob("module_*_FAI_*"))
    )
    for candidate in candidates:
        match = re.match(r"module_(\d+)_FAI_", candidate.name)
        if not match:
            continue
        module_dir = page_dir / "stage2_fai" / f"module_{int(match.group(1)):03d}"
        observation = candidate / "round_01" / "observation.png"
        xray = module_dir / "xray.png"
        system_prompt = module_dir / "system_prompt_xray.txt"
        user_prompt = module_dir / "user_prompt.txt"
        if not all(path.is_file() for path in (observation, xray, system_prompt, user_prompt)):
            continue
        result_path = candidate / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
        fai_number = result.get("fai_number")
        if not fai_number:
            number_match = re.search(r"_FAI_(.+)_\d+$", candidate.name)
            fai_number = number_match.group(1) if number_match else None
        return {
            "config": config,
            "candidate_key": candidate.name,
            "fai_number": fai_number,
            "observation": observation,
            "xray": xray,
            "system_prompt": system_prompt.read_text(encoding="utf-8"),
            "user_prompt": user_prompt.read_text(encoding="utf-8"),
        }
    raise FileNotFoundError("No candidate has both saved Stage 2 and Stage 3 request inputs")


def active_local_runs() -> list[str]:
    database = ROOT / "web_data" / "claw_view.sqlite3"
    if not database.is_file():
        return []
    with sqlite3.connect(database) as connection:
        return [
            row[0]
            for row in connection.execute(
                "SELECT id FROM runs WHERE status IN ('rendering', 'processing')"
            )
        ]


def status_from_error(error: str) -> int | None:
    match = HTTP_STATUS_RE.search(error)
    return int(match.group(1)) if match else None


def measured_call(role: str, request: Callable[[], dict[str, Any]], started: Event | None = None) -> dict[str, Any]:
    begin = time.monotonic()
    wall_time = datetime.now(timezone.utc).isoformat()
    if started is not None:
        started.set()
    try:
        outcome = request()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        outcome = {
            "http_status": getattr(exc, "status_code", None) or status_from_error(error),
            "error": error[:500],
        }
    end = time.monotonic()
    return {
        "role": role,
        "started_at_utc": wall_time,
        "start_monotonic": begin,
        "end_monotonic": end,
        "duration_seconds": round(end - begin, 3),
        **outcome,
    }


def run_condition(
    condition: str,
    stage2_request: Callable[[], dict[str, Any]],
    stage3_request: Callable[[], dict[str, Any]],
    overlap_delay: float,
) -> dict[str, Any]:
    if condition == "serial":
        stage2 = measured_call("stage2", stage2_request)
        stage3 = measured_call("stage3", stage3_request)
    else:
        stage2_started = Event()
        with ThreadPoolExecutor(max_workers=2) as executor:
            stage2_future = executor.submit(measured_call, "stage2", stage2_request, stage2_started)
            if not stage2_started.wait(timeout=10):
                raise RuntimeError("Stage 2 worker did not start")
            time.sleep(overlap_delay)
            stage3_future = executor.submit(measured_call, "stage3", stage3_request)
            stage2 = stage2_future.result()
            stage3 = stage3_future.result()
    overlap = max(
        0.0,
        min(stage2["end_monotonic"], stage3["end_monotonic"])
        - max(stage2["start_monotonic"], stage3["start_monotonic"]),
    )
    return {
        "condition": condition,
        "overlap_seconds": round(overlap, 3),
        "stage2": stage2,
        "stage3": stage3,
    }


def assess(conditions: list[dict[str, Any]], trials: int) -> str:
    serial = [item for item in conditions if item["condition"] == "serial"]
    overlapping = [item for item in conditions if item["condition"] == "overlap"]
    serial_success = all(item["stage3"].get("http_status") == 200 for item in serial)
    overlapped = [item for item in overlapping if item["overlap_seconds"] > 0]
    concurrent_500 = sum(item["stage3"].get("http_status") == 500 for item in overlapped)
    if len(serial) != trials or len(overlapped) != trials:
        return "INCONCLUSIVE: not all planned calls ran or overlapped"
    if not serial_success:
        return "NOT ISOLATED: Stage 3 also fails during serial calls; concurrency alone cannot explain it"
    if concurrent_500 == trials:
        return "REPRODUCED: Stage 3 succeeded in every serial trial and returned HTTP 500 in every overlapping trial"
    if concurrent_500:
        return "PARTIAL: Stage 3 only returned HTTP 500 in some overlapping trials; repeat and inspect server logs"
    return "NOT REPRODUCED: no Stage 3 HTTP 500 appeared during overlapping trials"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-dir", type=Path, required=True, help="Existing pipeline/page_XXXX directory")
    parser.add_argument("--candidate-key", help="Stage 3 candidate key; defaults to the first usable candidate")
    parser.add_argument("--trials", type=int, default=3, help="Serial/overlap pairs (default: 3)")
    parser.add_argument("--overlap-delay", type=float, default=2.0, help="Seconds after starting Stage 2 before Stage 3 begins")
    parser.add_argument("--cooldown", type=float, default=2.0, help="Seconds between conditions")
    parser.add_argument("--output", type=Path, help="JSON report path; defaults to a temporary file")
    parser.add_argument("--ignore-local-runs", action="store_true", help="Run even if the local app reports an active pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Show chosen inputs without calling the server")
    args = parser.parse_args()
    if args.trials < 1 or args.overlap_delay < 0 or args.cooldown < 0:
        parser.error("trials must be positive and delays must be non-negative")

    page_dir = args.page_dir.expanduser().resolve()
    selected = pick_inputs(page_dir, args.candidate_key)
    config = selected["config"]
    stage2_endpoint = config["endpoint"]
    stage3_endpoint = config["recovery_endpoint"]
    stage2_model = config["front_model"]
    stage3_model = config["recovery_model"]
    if (stage2_endpoint, stage2_model) != (stage3_endpoint, stage3_model):
        parser.error("This experiment requires Stage 2 and Stage 3 to use the same endpoint and model")

    schema_enabled = schema_parameter_enabled()
    proxy_url = os.environ.get("CLAW_VIEW_PROXY_URL") or get_enabled_proxy_url() or ""
    print(f"Endpoint: {stage2_endpoint}; model: {stage2_model}")
    print(f"Candidate: {selected['candidate_key']}; schema requested by current code: {schema_enabled}")
    print(f"Proxy: {'enabled' if proxy_url else 'disabled'}; trials: {args.trials}")
    if args.dry_run:
        print(f"Stage 2 image: {selected['xray']}")
        print(f"Stage 3 image: {selected['observation']}")
        return 0

    active = active_local_runs()
    if active and not args.ignore_local_runs:
        parser.error(
            "A local pipeline is active; stop or finish it before this controlled experiment. "
            "Use --ignore-local-runs only when its database status is stale."
        )
    if active:
        print("Warning: a local pipeline is marked active; outside traffic can confound this experiment")

    os.environ["CLAW_VIEW_PROXY_URL"] = proxy_url
    api_key = os.environ.get("CLAW_VIEW_API_KEY", pipeline.DEFAULT_API_KEY)
    stage2_client = pipeline.create_openai_client(base_url=stage2_endpoint, api_key=api_key, timeout=600)
    stage3_client = pipeline.create_openai_client(base_url=stage3_endpoint, api_key=api_key, timeout=600)
    with Image.open(selected["xray"]) as source:
        xray = source.convert("RGB")
    with Image.open(selected["observation"]) as source:
        observation = source.convert("RGB")

    def stage2_request() -> dict[str, Any]:
        raw, metadata = pipeline.call_vision_once(
            stage2_client, stage2_model, selected["system_prompt"], selected["user_prompt"],
            xray, int(config["fai_max_tokens"]), float(config["temperature"]),
        )
        return {"http_status": 200, "finish_reason": metadata.get("finish_reason"), "response_chars": len(raw)}

    def stage3_request() -> dict[str, Any]:
        decision = pipeline.request_recovery_decision(
            stage3_client, stage3_model, observation,
            round_number=1, fai_number=selected["fai_number"],
            max_tokens=int(config["recovery_max_tokens"]),
            temperature=float(config["recovery_temperature"]),
            step_norm=int(config["recovery_step_norm"]),
        )
        return {
            "http_status": 200 if decision.metadata else status_from_error(decision.error or ""),
            "action": decision.action,
            "decision_error": (decision.error or "")[:500] or None,
            "response_chars": len(decision.raw_response),
        }

    conditions: list[dict[str, Any]] = []
    try:
        for trial in range(1, args.trials + 1):
            order = ("serial", "overlap") if trial % 2 else ("overlap", "serial")
            for condition in order:
                print(f"Trial {trial}/{args.trials}: {condition} ...", flush=True)
                result = run_condition(condition, stage2_request, stage3_request, args.overlap_delay)
                result["trial"] = trial
                conditions.append(result)
                print(
                    f"  Stage 2 HTTP {result['stage2'].get('http_status')}; "
                    f"Stage 3 HTTP {result['stage3'].get('http_status')}; "
                    f"overlap {result['overlap_seconds']:.3f}s",
                    flush=True,
                )
                if len(conditions) < args.trials * 2:
                    time.sleep(args.cooldown)
    finally:
        stage2_client.close()
        stage3_client.close()

    conclusion = assess(conditions, args.trials)
    report = {
        "page_dir": str(page_dir),
        "candidate_key": selected["candidate_key"],
        "endpoint": stage2_endpoint,
        "model": stage2_model,
        "proxy_enabled": bool(proxy_url),
        "schema_parameter_enabled": schema_enabled,
        "trials": args.trials,
        "conditions": conditions,
        "conclusion": conclusion,
    }
    output = args.output or Path(tempfile.gettempdir()) / f"vision_stage3_concurrency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(conclusion)
    print(f"Report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
