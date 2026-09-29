#!/usr/bin/env python3
"""Diagnose Qwen 3.6 35B image extraction and OpenAI-compatible request options.

Run from the project root:
    python3 tests/diagnose_qwen35b_extraction.py

The default input is the most recently failed 35B extraction crop from the
local SQLite database. Pass --image to use a different screenshot. This tool
never updates the production database.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import platform
import sqlite3
import sys
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web_backend.extraction import DEFAULT_MODEL, PROMPT_VERSION, SYSTEM_PROMPT, _parse_response  # noqa: E402


DEFAULT_BASE_URL = "http://127.0.0.1:8001/v1"
USER_PROMPT = "请抽取这张检验项截图中有直接证据的字段。"
DEFAULT_CASES = ("production_json", "prompt_only", "json_no_thinking")
LOG_PATH: Path | None = None


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def log(message: str) -> None:
    line = f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    if LOG_PATH is not None:
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def default_image_from_latest_failed_record(database: Path, model: str) -> tuple[Path, dict[str, Any]]:
    if not database.is_file():
        raise FileNotFoundError(
            f"找不到本地任务数据库：{database}。请使用 --image 指定截图。"
        )
    uri = database.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        db.row_factory = sqlite3.Row
        row = db.execute(
            """SELECT id, run_id, fai, extraction_status, extraction_error,
                      extraction_model, model_crop_path, user_crop_path, updated_at
               FROM records
               WHERE extraction_status = 'failed'
                 AND extraction_model = ?
               ORDER BY updated_at DESC LIMIT 1""",
            (model,),
        ).fetchone()
    if row is None:
        raise FileNotFoundError(
            f"数据库中没有模型 {model!r} 的失败记录。请使用 --image 指定截图。"
        )
    record = dict(row)
    image = Path(record.get("user_crop_path") or record["model_crop_path"])
    if not image.is_file():
        raise FileNotFoundError(
            f"最新失败记录截图不存在：{image}（record_id={record['id']}）。请用 --image 指定截图。"
        )
    return image, record


def image_description(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    with Image.open(path) as image:
        width, height = image.size
        image_format = image.format
        mode = image.mode
    return {
        "path": str(path.resolve()),
        "file_name": path.name,
        "format": image_format,
        "mode": mode,
        "width": width,
        "height": height,
        "file_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "mime_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
    }


def response_content(envelope: dict[str, Any]) -> str | None:
    choices = envelope.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text", ""), str)
        )
    return None


def response_diagnostics(envelope: dict[str, Any], content: str | None) -> dict[str, Any]:
    choices = envelope.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice, dict) else {}
    message = message if isinstance(message, dict) else {}
    result: dict[str, Any] = {
        "response_model": envelope.get("model"),
        "response_id": envelope.get("id"),
        "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
        "usage": envelope.get("usage"),
        "assistant_role": message.get("role"),
        "assistant_refusal": message.get("refusal"),
        "content_present": bool(content),
        "content_chars": len(content or ""),
        "think_open_count": (content or "").count("<think>"),
        "think_close_count": (content or "").count("</think>"),
    }
    if content is None:
        result["json_parse"] = {"ok": False, "error": "No assistant message content found"}
        result["production_parser"] = {"ok": False, "error": "No assistant message content found"}
        return result

    try:
        fields, normalized, needs_review = _parse_response(content)
        result["production_parser"] = {
            "ok": True,
            "needs_review": needs_review,
            "recognized_fields": fields,
            "field_statuses": {name: item.get("status") for name, item in normalized.items()},
        }
    except Exception as exc:
        result["production_parser"] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    cleaned = content.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.rstrip().endswith("```"):
        cleaned = cleaned.rstrip()[:-3]
    cleaned = cleaned.strip()
    try:
        parsed = json.loads(cleaned)
        result["json_parse"] = {
            "ok": True,
            "root_type": type(parsed).__name__,
            "root_keys": list(parsed.keys()) if isinstance(parsed, dict) else None,
            "fields_type": type(parsed.get("fields")).__name__ if isinstance(parsed, dict) else None,
            "field_keys": list(parsed.get("fields", {}).keys()) if isinstance(parsed, dict) and isinstance(parsed.get("fields"), dict) else None,
        }
    except Exception as exc:
        result["json_parse"] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return result


def make_case(name: str) -> dict[str, Any]:
    case: dict[str, Any] = {"name": name, "response_format": None, "chat_template_kwargs": None}
    if name in {"production_json", "json_no_thinking"}:
        case["response_format"] = {"type": "json_object"}
    if name == "json_no_thinking":
        case["chat_template_kwargs"] = {"enable_thinking": False}
    return case


def redacted_payload(payload: dict[str, Any], image_info: dict[str, Any]) -> dict[str, Any]:
    safe = json.loads(json.dumps(payload))
    for message in safe.get("messages", []):
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "image_url":
                    part["image_url"]["url"] = (
                        f"<REDACTED image; sha256={image_info['sha256']}; "
                        f"bytes={image_info['file_bytes']}; mime={image_info['mime_type']}>"
                    )
    return safe


def call_case(
    *,
    case: dict[str, Any],
    image_data_uri: str,
    image_info: dict[str, Any],
    endpoint: str,
    model: str,
    api_key: str,
    timeout: float,
    output_dir: Path,
) -> dict[str, Any]:
    name = case["name"]
    case_dir = output_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                ],
            },
        ],
        "temperature": 0,
        "max_tokens": 1600,
    }
    if case["response_format"] is not None:
        payload["response_format"] = case["response_format"]
    if case["chat_template_kwargs"] is not None:
        payload["chat_template_kwargs"] = case["chat_template_kwargs"]

    request_path = f"{endpoint.rstrip('/')}/chat/completions"
    request_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    (case_dir / "request_redacted.json").write_text(
        json.dumps(redacted_payload(payload, image_info), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    request = urllib.request.Request(
        request_path,
        data=request_body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    log(f"CASE {name}: POST {request_path}; request_body_bytes={len(request_body)}; response_format={case['response_format']}; chat_template_kwargs={case['chat_template_kwargs']}")
    started = time.perf_counter()
    http_status: int | None = None
    response_headers: dict[str, str] = {}
    response_bytes = b""
    network_error: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            http_status = response.status
            response_headers = dict(response.headers.items())
            response_bytes = response.read()
    except urllib.error.HTTPError as exc:
        http_status = exc.code
        response_headers = dict(exc.headers.items()) if exc.headers else {}
        response_bytes = exc.read()
        network_error = f"HTTPError: {exc.code} {exc.reason}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        network_error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    response_text = response_bytes.decode("utf-8", errors="replace")
    (case_dir / "response_raw.txt").write_text(response_text, encoding="utf-8")

    try:
        envelope = json.loads(response_text) if response_text else None
        envelope_error = None
    except json.JSONDecodeError as exc:
        envelope = None
        envelope_error = f"JSONDecodeError: {exc}"
    if isinstance(envelope, dict):
        content = response_content(envelope)
        details = response_diagnostics(envelope, content)
        if content is not None:
            (case_dir / "assistant_content.txt").write_text(content, encoding="utf-8")
    else:
        content = None
        details = {"json_parse": {"ok": False, "error": envelope_error or "empty HTTP response"},
                   "production_parser": {"ok": False, "error": "No valid response envelope"}}

    summary = {
        "case": name,
        "requested_model": model,
        "request_url": request_path,
        "request_options": {
            "temperature": 0,
            "max_tokens": 1600,
            "response_format": case["response_format"],
            "chat_template_kwargs": case["chat_template_kwargs"],
        },
        "http_status": http_status,
        "response_headers": response_headers,
        "network_error": network_error,
        "response_envelope_json_error": envelope_error,
        "response_body_bytes": len(response_bytes),
        "elapsed_seconds": round(elapsed, 3),
        "image": image_info,
        "prompt_version": PROMPT_VERSION,
        **details,
    }
    (case_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(
        f"CASE {name} done: HTTP={http_status}, elapsed={elapsed:.2f}s, "
        f"finish_reason={summary.get('finish_reason')}, "
        f"usage={summary.get('usage')}, "
        f"json_ok={summary.get('json_parse', {}).get('ok')}, "
        f"production_parser_ok={summary.get('production_parser', {}).get('ok')}"
    )
    if network_error:
        log(f"CASE {name} error: {network_error}")
    if response_headers.get("Warning"):
        log(f"CASE {name} server Warning header: {response_headers['Warning']}")
    if envelope_error:
        log(f"CASE {name} invalid response envelope: {envelope_error}")
    if content is not None and not summary.get("json_parse", {}).get("ok"):
        log(f"CASE {name} assistant content was not valid JSON; see {case_dir / 'assistant_content.txt'}")
    return summary


def fetch_models(endpoint: str, api_key: str, timeout: float, output_dir: Path) -> dict[str, Any]:
    url = f"{endpoint.rstrip('/')}/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    started = time.perf_counter()
    result: dict[str, Any] = {"url": url, "http_status": None, "elapsed_seconds": None}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
            result.update(http_status=response.status, response_headers=dict(response.headers.items()))
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        result.update(http_status=exc.code, response_headers=dict(exc.headers.items()) if exc.headers else {}, error=f"HTTPError: {exc.code} {exc.reason}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raw = b""
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    text = raw.decode("utf-8", errors="replace")
    (output_dir / "models_response_raw.txt").write_text(text, encoding="utf-8")
    try:
        model_data = json.loads(text) if text else None
        result["response_json_valid"] = isinstance(model_data, (dict, list))
        result["available_model_ids"] = [item.get("id") for item in model_data.get("data", []) if isinstance(item, dict)] if isinstance(model_data, dict) else None
        result["requested_model_listed"] = None
    except json.JSONDecodeError as exc:
        result["response_json_valid"] = False
        result["json_error"] = f"JSONDecodeError: {exc}"
    (output_dir / "models_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="截图路径；默认读取最近一次 35B 失败记录的截图")
    parser.add_argument("--database", type=Path, default=PROJECT_ROOT / "web_data" / "claw_view.sqlite3")
    parser.add_argument("--base-url", default=os.environ.get("CLAW_VIEW_EXTRACTION_ENDPOINT") or os.environ.get("CLAW_VIEW_ENDPOINT", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("CLAW_VIEW_EXTRACTION_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-key", default=os.environ.get("CLAW_VIEW_API_KEY", "anything"), help="默认读取 CLAW_VIEW_API_KEY")
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("CLAW_VIEW_EXTRACTION_TIMEOUT", "180")))
    parser.add_argument("--output-dir", type=Path, help="诊断输出目录；默认 test_output/qwen35b_diagnostics/<timestamp>")
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES), help="逗号分隔: production_json,prompt_only,json_no_thinking")
    parser.add_argument("--skip-model-list", action="store_true", help="跳过 GET /models 检查")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or PROJECT_ROOT / "test_output" / "qwen35b_diagnostics" / utc_stamp()
    output_dir.mkdir(parents=True, exist_ok=True)
    global LOG_PATH
    LOG_PATH = output_dir / "debug.log"
    log(f"Diagnostic output: {output_dir.resolve()}")
    log(f"Python={sys.version.split()[0]} platform={platform.platform()}")
    log(f"Endpoint={args.base_url.rstrip('/')} model={args.model} timeout={args.timeout}s")

    record_info = None
    try:
        image_path = args.image.expanduser().resolve() if args.image else None
        if image_path is None:
            image_path, record_info = default_image_from_latest_failed_record(args.database, args.model)
        if not image_path.is_file():
            raise FileNotFoundError(f"找不到截图文件：{image_path}")
        image_info = image_description(image_path)
        image_bytes = image_path.read_bytes()
    except Exception as exc:
        log(f"INPUT ERROR: {type(exc).__name__}: {exc}")
        return 2

    run_metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "endpoint": args.base_url.rstrip("/"),
        "requested_model": args.model,
        "prompt_version": PROMPT_VERSION,
        "image": image_info,
        "source_record": record_info,
        "cases": [name.strip() for name in args.cases.split(",") if name.strip()],
        "api_key": "<redacted>",
        "database_mode": "read-only; no business data is written",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "system_prompt.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")
    (output_dir / "user_prompt.txt").write_text(USER_PROMPT, encoding="utf-8")
    log(f"Image={image_info['path']} size={image_info['width']}x{image_info['height']} bytes={image_info['file_bytes']} sha256={image_info['sha256']}")
    if record_info:
        log(f"Source failed record={record_info['id']} run={record_info['run_id']} fai={record_info['fai']} model={record_info['extraction_model']} prior_error={record_info['extraction_error']}")

    if not args.skip_model_list:
        models_summary = fetch_models(args.base_url, args.api_key, args.timeout, output_dir)
        ids = models_summary.get("available_model_ids") or []
        models_summary["requested_model_listed"] = args.model in ids if ids else None
        (output_dir / "models_summary.json").write_text(json.dumps(models_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"GET /models: HTTP={models_summary.get('http_status')}, requested_model_listed={models_summary.get('requested_model_listed')}, elapsed={models_summary.get('elapsed_seconds')}s")
        if models_summary.get("error"):
            log(f"GET /models error: {models_summary['error']}")

    mime_type = image_info["mime_type"]
    image_data_uri = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    case_summaries: list[dict[str, Any]] = []
    valid_cases = {"production_json", "prompt_only", "json_no_thinking"}
    requested_cases = [name.strip() for name in args.cases.split(",") if name.strip()]
    for name in requested_cases:
        if name not in valid_cases:
            log(f"SKIP unknown case: {name}")
            case_summaries.append({"case": name, "error": "unknown case"})
            continue
        try:
            case_summaries.append(
                call_case(
                    case=make_case(name),
                    image_data_uri=image_data_uri,
                    image_info=image_info,
                    endpoint=args.base_url,
                    model=args.model,
                    api_key=args.api_key,
                    timeout=args.timeout,
                    output_dir=output_dir,
                )
            )
        except Exception as exc:
            error = {"case": name, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
            case_summaries.append(error)
            (output_dir / f"{name}_client_exception.json").write_text(json.dumps(error, ensure_ascii=False, indent=2), encoding="utf-8")
            log(f"CASE {name} client exception: {error['error']}")

    final = {
        **run_metadata,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "case_summaries": [
            {
                "case": item.get("case"),
                "http_status": item.get("http_status"),
                "network_error": item.get("network_error", item.get("error")),
                "finish_reason": item.get("finish_reason"),
                "usage": item.get("usage"),
                "json_ok": item.get("json_parse", {}).get("ok"),
                "production_parser_ok": item.get("production_parser", {}).get("ok"),
                "elapsed_seconds": item.get("elapsed_seconds"),
            }
            for item in case_summaries
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    log("All requested diagnostic cases finished.")
    log(f"Summary file: {output_dir / 'summary.json'}")
    for item in final["case_summaries"]:
        log(
            f"RESULT {item['case']}: http={item['http_status']} "
            f"finish={item['finish_reason']} json={item['json_ok']} "
            f"app_parser={item['production_parser_ok']} elapsed={item['elapsed_seconds']}s"
        )
    failed = [item for item in case_summaries if item.get("http_status") != 200 or not item.get("production_parser", {}).get("ok")]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
