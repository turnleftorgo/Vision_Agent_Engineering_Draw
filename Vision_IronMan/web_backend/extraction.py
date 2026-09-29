from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .store import ScanStore


DEFAULT_ENDPOINT = "http://127.0.0.1:8001/v1"
DEFAULT_MODEL = "Qwen3.6-35B-A3B-MLX-8bit"
FIELDS = {
    "fai", "spc", "description", "nominal", "usl", "lsl",
    "hundred_percent", "dc", "points",
}
PROMPT_VERSION = "inspection-crop-v4-fai-marker-context"
SYSTEM_PROMPT = """你是机械工程图纸检验记录的视觉字段抽取器。只根据用户提供的这张检验截图识别字段，不推测图片中没有明确显示的内容。

必须只输出一个合法 JSON 对象，不要 Markdown、代码围栏、解释或额外字段。格式：
{"fields":{"fai":{"value":null,"status":"not_found","evidence":"","confidence":0.0}}}

fields 可包含以下任意键：fai、spc、description、nominal、usl、lsl、hundred_percent、dc、points；这些字段都是可选的，并非每张截图都会出现。没有证据的字段可以省略，也可以返回 value=null、status=not_found。每个实际返回的字段使用 value、status、evidence、confidence。value 仅为可见原文字符串或 null；status 只能是 found、not_found、unreadable、ambiguous；confidence 为 0 到 1。

字段定义：
- fai：与该检验要求关联的 FAI 编号/标记。
- spc：与该 FAI 同组的 SPC 代码/标记。仅在关联关系明确时填写。
- description：只填写与检验项关联的独立文字说明、特征名称或检验指令。单独的尺寸标注（例如“0.88 ±0.10”）属于 nominal/usl/lsl，不能复制到 description；没有独立文字说明时留空/ not_found。
- nominal：尺寸标注中的名义值/基本值。例如“0.88 ±0.10”中的 nominal 是“0.88”。
- usl、lsl：填写相对于 nominal 的有符号公差偏差，不填写计算后的绝对边界。对于“0.88 ±0.10”，必须输出 usl="+0.10"、lsl="-0.10"；严禁计算或输出“0.98”和“0.78”。对于“0.88 +0.05/-0.02”，必须分别原样输出 usl="+0.05"、lsl="-0.02"。只有单侧公差时，只填写图上明确给出的那一侧，另一侧为 not_found。若图纸直接标出明确的上/下偏差，按其上、下方向保留符号；不得把偏差加减到 nominal 上换算绝对极限值。只有无法辨认正负方向时才标为 ambiguous/unreadable，不得静默丢弃符号或把同一个无符号值同时填入 USL 和 LSL。
- hundred_percent：明确的 100% 检验标记或文字。
- dc：明确的 DC 标记。
- points：明确列出的测量点/位置编号或范围，保留原文。

关键规则：并非每张截图都包含上述所有字段，尤其 DC 可能完全不存在。未出现时可以省略该键，也可返回 value=null、status=not_found；出现但不可辨认时为 unreadable；无法确认它属于当前 FAI 时为 ambiguous。不要为了满足字段清单而虚构占位值；不要从常识推测、不要把表格上下文当成图片证据。公差栏严格记录带符号的偏差，绝不计算绝对极限值；尺寸数值不能重复塞入 description。只有 status=found 且 value 有直接视觉证据时才给出非空 value。evidence 用简短文字指出图中依据；无证据时为空字符串。"""


def _model_config() -> tuple[str, str, str]:
    endpoint = os.environ.get("CLAW_VIEW_EXTRACTION_ENDPOINT") or os.environ.get(
        "CLAW_VIEW_ENDPOINT", DEFAULT_ENDPOINT
    )
    model = os.environ.get("CLAW_VIEW_EXTRACTION_MODEL", DEFAULT_MODEL)
    api_key = os.environ.get("CLAW_VIEW_API_KEY", "anything")
    return endpoint.rstrip("/"), model, api_key


def _parse_response(content: str) -> tuple[dict[str, str], dict[str, Any], bool]:
    text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("模型没有返回 JSON 对象")
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict) or not isinstance(payload.get("fields"), dict):
        raise ValueError("模型 JSON 缺少 fields 对象")

    cleaned: dict[str, str] = {}
    normalized: dict[str, Any] = {}
    needs_review = False
    for name in FIELDS:
        item = payload["fields"].get(name)
        if name not in payload["fields"] or item is None:
            normalized[name] = {
                "value": None,
                "status": "not_found",
                "evidence": "",
                "confidence": 0.0,
            }
            continue
        if not isinstance(item, dict):
            normalized[name] = {
                "value": None,
                "status": "unreadable",
                "evidence": "",
                "confidence": 0.0,
            }
            needs_review = True
            continue
        status = item.get("status")
        if status not in {"found", "not_found", "unreadable", "ambiguous"}:
            status = "unreadable"
            needs_review = True
        raw_value = item.get("value")
        value = raw_value.strip() if isinstance(raw_value, str) else None
        if status == "found" and value:
            cleaned[name] = value[:2000]
        elif status in {"ambiguous", "unreadable"}:
            needs_review = True
        try:
            confidence = float(item.get("confidence") or 0)
        except (TypeError, ValueError):
            confidence = 0.0
        normalized[name] = {
            "value": value,
            "status": status,
            "evidence": str(item.get("evidence") or "")[:500],
            "confidence": max(0.0, min(1.0, confidence)),
        }
    return cleaned, normalized, needs_review


def _call_model(image_path: Path, endpoint: str, model: str, api_key: str) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    request_body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "请抽取这张检验项截图中有直接证据的字段。红色矩形框标出本条记录的 FAI 标记；"
                                "只提取与框内 FAI 及其同组 SPC 明确对应的参数。依据图中的引线、标注和空间关系确认归属，"
                                "不要把邻近其他 FAI/SPC 或其他尺寸项的内容并入本条记录。红框是定位提示，不是字段值来源；"
                                "所有字段仍须有图中直接证据。"
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                    ],
                },
            ],
            "temperature": 0,
            "max_tokens": 1600,
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{endpoint}/chat/completions",
        data=request_body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    timeout = float(os.environ.get("CLAW_VIEW_EXTRACTION_TIMEOUT", "180"))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read(1000).decode("utf-8", errors="replace")
        raise RuntimeError(f"Qwen 服务 HTTP {exc.code}: {detail}") from exc
    choices = result.get("choices") or []
    if not choices:
        raise RuntimeError("Qwen 服务未返回 choices")
    content = choices[0].get("message", {}).get("content")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Qwen 服务返回了空识别结果")
    return content


def _run_extraction(
    store: ScanStore, record: dict[str, Any], record_id: str,
    endpoint: str, model: str, api_key: str,
) -> None:
    version = int(record["crop_version"])
    digest = str(record["crop_sha256"])
    if not store.set_extraction_status(record_id, version, "processing"):
        return
    try:
        image_path = Path(record["effective_crop_path"])
        content = _call_model(image_path, endpoint, model, api_key)
        fields, normalized, needs_review = _parse_response(content)
        result = {
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "crop_version": version,
            "fields": normalized,
        }
        applied = store.apply_extraction(
            record_id, version, digest, fields, result,
            "needs_review" if needs_review else "completed",
        )
        if applied:
            print(
                f"[EXTRACTION][{record_id}] {model} complete "
                f"(crop v{version}, {len(fields)} visible field(s))",
                flush=True,
            )
    except Exception as exc:
        store.set_extraction_status(
            record_id, version, "failed", error=f"{type(exc).__name__}: {exc}"
        )
        print(
            f"[EXTRACTION][{record_id}] failed: {type(exc).__name__}: {exc}",
            flush=True,
        )


_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen-vl-extract")
_FUTURES: set[Future[Any]] = set()


def queue_record_extraction(store: ScanStore, run_id: str, record_id: str) -> bool:
    endpoint, model, api_key = _model_config()
    record = store.get_record(run_id, record_id)
    if not record:
        return False
    image_path = Path(record["effective_crop_path"])
    if not image_path.is_file():
        store.set_extraction_status(
            record_id, int(record.get("crop_version") or 0), "failed",
            error="有效截图文件不存在",
        )
        return False
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    prepared = store.prepare_extraction(record_id, digest, model)
    if prepared is None:
        return False
    future = _EXECUTOR.submit(
        _run_extraction, store, prepared, record_id, endpoint, model, api_key
    )
    _FUTURES.add(future)
    future.add_done_callback(_FUTURES.discard)
    return True
