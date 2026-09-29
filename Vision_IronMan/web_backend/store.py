from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ScanStore:
    """SQLite persistence with nullable columns reserved for future extraction."""

    def __init__(self, database_path: Path, backend_instance_id: str | None = None) -> None:
        self.database_path = database_path
        self.backend_instance_id = backend_instance_id or uuid.uuid4().hex
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    pdf_name TEXT NOT NULL,
                    pdf_path TEXT NOT NULL,
                    run_dir TEXT NOT NULL,
                    status TEXT NOT NULL,
                    total_pages INTEGER NOT NULL DEFAULT 0,
                    current_page INTEGER NOT NULL DEFAULT 0,
                    current_module INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    project TEXT,
                    revision TEXT,
                    author TEXT,
                    drawing_date TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS records (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    candidate_key TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    project TEXT,
                    revision TEXT,
                    author TEXT,
                    drawing_date TEXT,
                    module TEXT,
                    module_index INTEGER,
                    page_index INTEGER NOT NULL,
                    fai TEXT,
                    spc TEXT,
                    description TEXT,
                    nominal TEXT,
                    usl TEXT,
                    lsl TEXT,
                    hundred_percent TEXT,
                    dc TEXT,
                    points TEXT,
                    page_width INTEGER NOT NULL,
                    page_height INTEGER NOT NULL,
                    model_bbox TEXT NOT NULL,
                    user_bbox TEXT,
                    model_crop_path TEXT NOT NULL,
                    user_crop_path TEXT,
                    marker_bbox TEXT,
                    source_page_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(run_id, page_index, candidate_key),
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                );
                CREATE TABLE IF NOT EXISTS crop_revisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id TEXT NOT NULL,
                    bbox TEXT NOT NULL,
                    crop_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(record_id) REFERENCES records(id)
                );
                CREATE INDEX IF NOT EXISTS idx_records_run ON records(run_id);
                CREATE TABLE IF NOT EXISTS page_progress (
                    run_id TEXT NOT NULL,
                    page_index INTEGER NOT NULL,
                    page_width INTEGER NOT NULL,
                    page_height INTEGER NOT NULL,
                    overview_path TEXT,
                    stage1_modules TEXT NOT NULL DEFAULT '[]',
                    stage2_modules TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY(run_id, page_index),
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                );
                """
            )
            record_columns = {
                row[1] for row in db.execute("PRAGMA table_info(records)").fetchall()
            }
            if "module_index" not in record_columns:
                db.execute("ALTER TABLE records ADD COLUMN module_index INTEGER")
            run_columns = {
                row[1] for row in db.execute("PRAGMA table_info(runs)").fetchall()
            }
            if "current_module" not in run_columns:
                db.execute(
                    "ALTER TABLE runs ADD COLUMN current_module INTEGER NOT NULL DEFAULT 0"
                )
            for column in ("project", "revision", "author", "drawing_date"):
                if column not in run_columns:
                    db.execute(f"ALTER TABLE runs ADD COLUMN {column} TEXT")
            if "backend_instance_id" not in run_columns:
                db.execute("ALTER TABLE runs ADD COLUMN backend_instance_id TEXT")
            db.execute(
                """UPDATE runs SET status = 'interrupted',
                    error = 'Backend process restarted; this task cannot be resumed.',
                    updated_at = ?
                WHERE status IN ('queued', 'rendering', 'processing')
                  AND (backend_instance_id IS NULL OR backend_instance_id != ?)""",
                (utc_now(), self.backend_instance_id),
            )

    def create_run(self, run_id: str, pdf_name: str, pdf_path: Path, run_dir: Path) -> None:
        now = utc_now()
        with self.connect() as db:
            db.execute(
                """INSERT INTO runs
                (id, pdf_name, pdf_path, run_dir, status, backend_instance_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'queued', ?, ?, ?)""",
                (run_id, pdf_name, str(pdf_path), str(run_dir), self.backend_instance_id, now, now),
            )

    def update_run(self, run_id: str, **fields: Any) -> None:
        allowed = {
            "status",
            "total_pages",
            "current_page",
            "current_module",
            "error",
        }
        values = {key: value for key, value in fields.items() if key in allowed}
        values["updated_at"] = utc_now()
        assignments = ", ".join(f"{key} = ?" for key in values)
        with self.connect() as db:
            db.execute(
                f"UPDATE runs SET {assignments} WHERE id = ?",
                (*values.values(), run_id),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def upsert_refined_record(self, run_id: str, event: dict[str, Any]) -> str:
        record_id = f"{run_id}-p{event['page_index']}-{event['candidate_key']}"
        now = utc_now()
        with self.connect() as db:
            metadata = db.execute(
                "SELECT project, revision, author, drawing_date FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            sequence = db.execute(
                "SELECT COUNT(*) FROM records WHERE run_id = ?", (run_id,)
            ).fetchone()[0] + 1
            db.execute(
                """INSERT INTO records (
                    id, run_id, candidate_key, sequence,
                    project, revision, author, drawing_date, module, module_index,
                    page_index, fai, spc, description,
                    nominal, usl, lsl, hundred_percent, dc, points,
                    page_width, page_height, model_bbox, model_crop_path,
                    marker_bbox, source_page_path, created_at, updated_at
                ) VALUES (
                    ?, ?, ?, ?,
                    ?, ?, ?, ?, NULL, ?,
                    ?, ?, ?, ?,
                    NULL, NULL, NULL, NULL, NULL, NULL,
                    ?, ?, ?, ?, ?, ?, ?, ?
                )
                ON CONFLICT(run_id, page_index, candidate_key) DO UPDATE SET
                    module_index = excluded.module_index,
                    fai = excluded.fai,
                    spc = excluded.spc,
                    description = excluded.description,
                    page_width = excluded.page_width,
                    page_height = excluded.page_height,
                    model_bbox = excluded.model_bbox,
                    model_crop_path = excluded.model_crop_path,
                    marker_bbox = excluded.marker_bbox,
                    source_page_path = excluded.source_page_path,
                    updated_at = excluded.updated_at
                """,
                (
                    record_id,
                    run_id,
                    event["candidate_key"],
                    sequence,
                    *(tuple(metadata) if metadata else (None, None, None, None)),
                    int(event.get("module_index") or 0),
                    int(event["page_index"]),
                    event.get("fai_number"),
                    event.get("spc_code"),
                    event.get("description") or event.get("target_summary") or "",
                    int(event["page_width"]),
                    int(event["page_height"]),
                    json.dumps(event["final_bbox_full_image"]),
                    event["refined_crop_path_absolute"],
                    json.dumps(event.get("marker_bbox_full_image")),
                    event["page_image"],
                    now,
                    now,
                ),
            )
        return record_id

    def save_stage1_progress(self, run_id: str, event: dict[str, Any]) -> None:
        with self.connect() as db:
            db.execute(
                """INSERT INTO page_progress(
                    run_id, page_index, page_width, page_height,
                    overview_path, stage1_modules, stage2_modules
                ) VALUES (?, ?, ?, ?, ?, ?, '{}')
                ON CONFLICT(run_id, page_index) DO UPDATE SET
                    page_width = excluded.page_width,
                    page_height = excluded.page_height,
                    overview_path = excluded.overview_path,
                    stage1_modules = excluded.stage1_modules
                """,
                (
                    run_id,
                    int(event["page_index"]),
                    int(event["page_width"]),
                    int(event["page_height"]),
                    event.get("visualization_path"),
                    json.dumps(event.get("modules", [])),
                ),
            )

    def save_stage2_progress(self, run_id: str, event: dict[str, Any]) -> None:
        page_index = int(event["page_index"])
        module_index = str(int(event["module_index"]))
        with self.connect() as db:
            row = db.execute(
                "SELECT stage2_modules FROM page_progress WHERE run_id = ? AND page_index = ?",
                (run_id, page_index),
            ).fetchone()
            modules = json.loads(row[0]) if row else {}
            modules[module_index] = {
                "module_index": int(event["module_index"]),
                "module_bbox_full_image": event["module_bbox_full_image"],
                "module_width": int(event["module_width"]),
                "module_height": int(event["module_height"]),
                "valid_cluster_count": int(event.get("valid_cluster_count", 0)),
                "valid_clusters": event.get("valid_clusters", []),
                "visualization_path": event.get("visualization_path"),
            }
            if row:
                db.execute(
                    """UPDATE page_progress SET stage2_modules = ?
                    WHERE run_id = ? AND page_index = ?""",
                    (json.dumps(modules), run_id, page_index),
                )
            else:
                db.execute(
                    """INSERT INTO page_progress(
                        run_id, page_index, page_width, page_height, stage2_modules
                    ) VALUES (?, ?, ?, ?, ?)""",
                    (
                        run_id,
                        page_index,
                        int(event["page_width"]),
                        int(event["page_height"]),
                        json.dumps(modules),
                    ),
                )

    def get_progress(self, run_id: str) -> dict[str, Any]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM page_progress WHERE run_id = ? ORDER BY page_index",
                (run_id,),
            ).fetchall()
            refined_count = db.execute(
                "SELECT COUNT(*) FROM records WHERE run_id = ?", (run_id,)
            ).fetchone()[0]
        pages: list[dict[str, Any]] = []
        module_count = 0
        candidate_count = 0
        for raw in rows:
            row = dict(raw)
            stage1_modules = json.loads(row["stage1_modules"] or "[]")
            stage2_modules = json.loads(row["stage2_modules"] or "{}")
            module_count += len(stage1_modules)
            candidate_count += sum(
                int(item.get("valid_cluster_count", 0))
                for item in stage2_modules.values()
            )
            pages.append(
                {
                    "page_index": row["page_index"],
                    "page_width": row["page_width"],
                    "page_height": row["page_height"],
                    "has_overview": bool(row["overview_path"]),
                    "stage1_modules": stage1_modules,
                    "stage2_modules": stage2_modules,
                }
            )
        percent = round(refined_count / candidate_count * 100) if candidate_count else 0
        return {
            "stage1_module_count": module_count,
            "candidate_count": candidate_count,
            "refined_count": refined_count,
            "percent": min(100, percent),
            "pages": pages,
        }

    def list_records(self, run_id: str) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM records WHERE run_id = ?", (run_id,)
            ).fetchall()
        records = [self._serialize_record(dict(row)) for row in rows]

        def fai_key(item: dict[str, Any]) -> tuple[int, float | str, int]:
            value = (item.get("fai") or "").strip()
            try:
                return (0, float(value), item["sequence"])
            except ValueError:
                return (1, value.lower(), item["sequence"])

        return sorted(records, key=fai_key)

    def get_record(self, run_id: str, record_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM records WHERE run_id = ? AND id = ?",
                (run_id, record_id),
            ).fetchone()
        return self._serialize_record(dict(row)) if row else None

    def update_record_fields(self, run_id: str, record_id: str, fields: dict[str, Any]) -> dict[str, Any] | None:
        allowed = {
            "project", "revision", "author", "drawing_date", "module", "fai",
            "spc", "description", "nominal", "usl", "lsl", "hundred_percent",
            "dc", "points",
        }
        values = {key: value for key, value in fields.items() if key in allowed}
        if not values:
            raise ValueError("没有可更新的检验字段")
        values["updated_at"] = utc_now()
        assignments = ", ".join(f"{key} = ?" for key in values)
        with self.connect() as db:
            db.execute(
                f"UPDATE records SET {assignments} WHERE run_id = ? AND id = ?",
                (*values.values(), run_id, record_id),
            )
        return self.get_record(run_id, record_id)

    def update_run_metadata(self, run_id: str, field: str, value: str | None) -> None:
        allowed = {"project", "revision", "author", "drawing_date"}
        if field not in allowed:
            raise ValueError("不允许修改该图纸资料字段")
        with self.connect() as db:
            db.execute(
                f"UPDATE runs SET {field} = ? WHERE id = ?",
                (value, run_id),
            )
            db.execute(
                f"UPDATE records SET {field} = ?, updated_at = ? WHERE run_id = ?",
                (value, utc_now(), run_id),
            )

    def save_user_crop(
        self, record_id: str, bbox: list[int], crop_path: Path
    ) -> None:
        now = utc_now()
        serialized = json.dumps(bbox)
        with self.connect() as db:
            db.execute(
                """UPDATE records
                SET user_bbox = ?, user_crop_path = ?, updated_at = ?
                WHERE id = ?""",
                (serialized, str(crop_path), now, record_id),
            )
            db.execute(
                """INSERT INTO crop_revisions(record_id, bbox, crop_path, created_at)
                VALUES (?, ?, ?, ?)""",
                (record_id, serialized, str(crop_path), now),
            )

    @staticmethod
    def _serialize_record(row: dict[str, Any]) -> dict[str, Any]:
        for key in ("model_bbox", "user_bbox", "marker_bbox"):
            row[key] = json.loads(row[key]) if row.get(key) else None
        row["effective_bbox"] = row["user_bbox"] or row["model_bbox"]
        row["effective_crop_path"] = row["user_crop_path"] or row["model_crop_path"]
        row["user_override"] = row["user_bbox"] is not None
        return row
