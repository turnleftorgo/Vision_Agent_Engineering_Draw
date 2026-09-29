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
                    project_user_set INTEGER NOT NULL DEFAULT 0,
                    revision_user_set INTEGER NOT NULL DEFAULT 0,
                    author_user_set INTEGER NOT NULL DEFAULT 0,
                    drawing_date_user_set INTEGER NOT NULL DEFAULT 0,
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
                    crop_sha256 TEXT,
                    crop_version INTEGER NOT NULL DEFAULT 0,
                    extraction_status TEXT NOT NULL DEFAULT 'not_started',
                    extraction_error TEXT,
                    extraction_result TEXT,
                    extraction_model TEXT,
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
                CREATE TABLE IF NOT EXISTS active_refinements (
                    run_id TEXT NOT NULL,
                    page_index INTEGER NOT NULL,
                    candidate_key TEXT NOT NULL,
                    module_index INTEGER,
                    fai_number TEXT,
                    round_number INTEGER NOT NULL DEFAULT 0,
                    bbox TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(run_id, page_index, candidate_key),
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                );
                """
            )
            record_columns = {
                row[1] for row in db.execute("PRAGMA table_info(records)").fetchall()
            }
            if "module_index" not in record_columns:
                db.execute("ALTER TABLE records ADD COLUMN module_index INTEGER")
            for column, declaration in (
                ("crop_sha256", "TEXT"),
                ("crop_version", "INTEGER NOT NULL DEFAULT 0"),
                ("extraction_status", "TEXT NOT NULL DEFAULT 'not_started'"),
                ("extraction_error", "TEXT"),
                ("extraction_result", "TEXT"),
                ("extraction_model", "TEXT"),
            ):
                if column not in record_columns:
                    db.execute(f"ALTER TABLE records ADD COLUMN {column} {declaration}")
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
            for column in (
                "project_user_set", "revision_user_set", "author_user_set",
                "drawing_date_user_set",
            ):
                if column not in run_columns:
                    db.execute(
                        f"ALTER TABLE runs ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0"
                    )
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
            if values.get("status") in {"completed", "failed", "interrupted"}:
                db.execute("DELETE FROM active_refinements WHERE run_id = ?", (run_id,))

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
                    None,
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

    def update_refinement(self, run_id: str, event: dict[str, Any]) -> None:
        page_index = int(event["page_index"])
        candidate_key = str(event["candidate_key"])
        with self.connect() as db:
            if event["type"] == "refinement_finished":
                db.execute(
                    "DELETE FROM active_refinements WHERE run_id = ? AND page_index = ? AND candidate_key = ?",
                    (run_id, page_index, candidate_key),
                )
                return
            bbox = event.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                return
            db.execute(
                """INSERT INTO active_refinements (
                    run_id, page_index, candidate_key, module_index, fai_number,
                    round_number, bbox, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, page_index, candidate_key) DO UPDATE SET
                    round_number = excluded.round_number,
                    bbox = excluded.bbox,
                    updated_at = excluded.updated_at
                WHERE excluded.round_number >= active_refinements.round_number""",
                (
                    run_id, page_index, candidate_key,
                    event.get("module_index"), event.get("fai_number"),
                    int(event.get("round") or 0), json.dumps(bbox), utc_now(),
                ),
            )

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
                "module_total": int(event.get("module_total") or 0),
                "module_bbox_full_image": event["module_bbox_full_image"],
                "module_width": int(event["module_width"]),
                "module_height": int(event["module_height"]),
                "valid_cluster_count": int(event.get("valid_cluster_count", 0)),
                "valid_clusters": event.get("valid_clusters", []),
                "page_metadata": event.get("page_metadata"),
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
            active_rows = db.execute(
                "SELECT * FROM active_refinements WHERE run_id = ? ORDER BY page_index, candidate_key",
                (run_id,),
            ).fetchall()
        refinements_by_page: dict[int, list[dict[str, Any]]] = {}
        for raw in active_rows:
            active = dict(raw)
            active["bbox"] = json.loads(active["bbox"])
            refinements_by_page.setdefault(active["page_index"], []).append(active)
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
                    "active_refinements": refinements_by_page.get(row["page_index"], []),
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
        lock_field = f"{field}_user_set"
        with self.connect() as db:
            db.execute(
                f"UPDATE runs SET {field} = ?, {lock_field} = 1 WHERE id = ?",
                (value, run_id),
            )
            db.execute(
                f"UPDATE records SET {field} = ?, updated_at = ? WHERE run_id = ?",
                (value, utc_now(), run_id),
            )

    def consolidate_page1_metadata(
        self, run_id: str, module_total: int
    ) -> dict[str, Any]:
        """Fill top-level metadata only from consistent, evidenced module candidates."""
        def candidate_value(module: dict[str, Any], field: str) -> str | None:
            metadata = module.get("page_metadata") or {}
            item = metadata.get(field) if isinstance(metadata, dict) else None
            if not isinstance(item, dict):
                return None
            value, evidence = item.get("value"), item.get("evidence")
            if (
                item.get("status") == "found"
                and isinstance(value, str) and value.strip()
                and isinstance(evidence, str) and evidence.strip()
            ):
                return value.strip()
            return None

        def normalized(value: str) -> str:
            return " ".join(value.split()).casefold()

        with self.connect() as db:
            progress = db.execute(
                "SELECT stage2_modules FROM page_progress WHERE run_id = ? AND page_index = 1",
                (run_id,),
            ).fetchone()
            if not progress:
                return {"status": "collecting", "filled": [], "conflicts": []}
            modules = json.loads(progress["stage2_modules"] or "{}")
            if module_total <= 0 or len(modules) < module_total:
                return {"status": "collecting", "filled": [], "conflicts": []}
            run = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if not run:
                return {"status": "missing_run", "filled": [], "conflicts": []}

            filled: list[str] = []
            conflicts: list[str] = []

            def fill_if_empty(db_field: str, value: str) -> bool:
                lock_field = f"{db_field}_user_set"
                if run[db_field] is not None and str(run[db_field]).strip():
                    return False
                if int(run[lock_field] or 0):
                    return False
                cursor = db.execute(
                    f"""UPDATE runs SET {db_field} = ?
                    WHERE id = ? AND {lock_field} = 0
                    AND ({db_field} IS NULL OR TRIM({db_field}) = '')""",
                    (value, run_id),
                )
                if not cursor.rowcount:
                    return False
                db.execute(
                    f"""UPDATE records SET {db_field} = ?, updated_at = ?
                    WHERE run_id = ? AND ({db_field} IS NULL OR TRIM({db_field}) = '')""",
                    (value, utc_now(), run_id),
                )
                return True

            project_values = {
                normalized(value): value
                for module in modules.values()
                if (value := candidate_value(module, "project")) is not None
            }
            if len(project_values) > 1:
                conflicts.append("project")
            elif project_values:
                project_value = next(iter(project_values.values()))
                if fill_if_empty("project", project_value):
                    filled.append("project")

            # These values describe one revision-history row and must never be
            # assembled independently from different module candidates.
            row_candidates: dict[tuple[str, str, str], tuple[str, str, str]] = {}
            for module in modules.values():
                revision = candidate_value(module, "revision")
                author = candidate_value(module, "author")
                date = candidate_value(module, "date")
                if revision and author and date:
                    key = (normalized(revision), normalized(author), normalized(date))
                    row_candidates[key] = (revision, author, date)
            if len(row_candidates) > 1:
                conflicts.append("revision_author_date")
            elif row_candidates:
                revision, author, date = next(iter(row_candidates.values()))
                row_values = {
                    "revision": revision,
                    "author": author,
                    "drawing_date": date,
                }
                has_mismatch = any(
                    run[field] is not None
                    and str(run[field]).strip()
                    and normalized(str(run[field])) != normalized(value)
                    for field, value in row_values.items()
                )
                if has_mismatch:
                    conflicts.append("revision_author_date")
                else:
                    for field, value in row_values.items():
                        if fill_if_empty(field, value):
                            filled.append(field)
            return {
                "status": "complete",
                "filled": filled,
                "conflicts": conflicts,
            }

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

    def prepare_extraction(
        self, record_id: str, crop_sha256: str, model: str
    ) -> dict[str, Any] | None:
        """Create a new extraction version only when effective crop pixels change."""
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM records WHERE id = ?", (record_id,)
            ).fetchone()
            if not row or row["crop_sha256"] == crop_sha256:
                return None
            version = int(row["crop_version"] or 0) + 1
            db.execute(
                """UPDATE records SET crop_sha256 = ?, crop_version = ?,
                    extraction_status = 'queued', extraction_error = NULL,
                    extraction_result = NULL, extraction_model = ?, updated_at = ?
                WHERE id = ?""",
                (crop_sha256, version, model, utc_now(), record_id),
            )
            result = dict(row)
            result.update(
                crop_sha256=crop_sha256,
                crop_version=version,
                extraction_status="queued",
                extraction_model=model,
                effective_crop_path=row["user_crop_path"] or row["model_crop_path"],
            )
            return result

    def set_extraction_status(
        self, record_id: str, version: int, status: str, *,
        error: str | None = None, result: dict[str, Any] | None = None,
    ) -> bool:
        with self.connect() as db:
            cursor = db.execute(
                """UPDATE records SET extraction_status = ?, extraction_error = ?,
                    extraction_result = ?, updated_at = ?
                WHERE id = ? AND crop_version = ?""",
                (
                    status,
                    error,
                    json.dumps(result, ensure_ascii=False) if result is not None else None,
                    utc_now(),
                    record_id,
                    version,
                ),
            )
            return cursor.rowcount == 1

    def apply_extraction(
        self, record_id: str, version: int, crop_sha256: str,
        fields: dict[str, str], result: dict[str, Any], status: str,
    ) -> bool:
        """Apply extracted values only to blank fields and only for current crop."""
        allowed = {
            "fai", "spc", "description", "nominal", "usl", "lsl",
            "hundred_percent", "dc", "points",
        }
        now = utc_now()
        with self.connect() as db:
            row = db.execute(
                "SELECT crop_sha256, crop_version FROM records WHERE id = ?",
                (record_id,),
            ).fetchone()
            if not row or row["crop_version"] != version or row["crop_sha256"] != crop_sha256:
                return False
            current = db.execute(
                "SELECT * FROM records WHERE id = ?", (record_id,)
            ).fetchone()
            updates = {
                key: value for key, value in fields.items()
                if key in allowed and value and not current[key]
            }
            assignments = [f"{key} = ?" for key in updates]
            assignments.extend([
                "extraction_status = ?", "extraction_error = NULL",
                "extraction_result = ?", "updated_at = ?",
            ])
            db.execute(
                f"UPDATE records SET {', '.join(assignments)} WHERE id = ? AND crop_version = ?",
                (
                    *updates.values(), status,
                    json.dumps(result, ensure_ascii=False), now,
                    record_id, version,
                ),
            )
            return True

    @staticmethod
    def _serialize_record(row: dict[str, Any]) -> dict[str, Any]:
        for key in ("model_bbox", "user_bbox", "marker_bbox"):
            row[key] = json.loads(row[key]) if row.get(key) else None
        row["effective_bbox"] = row["user_bbox"] or row["model_bbox"]
        row["effective_crop_path"] = row["user_crop_path"] or row["model_crop_path"]
        row["user_override"] = row["user_bbox"] is not None
        if row.get("extraction_result"):
            try:
                row["extraction_result"] = json.loads(row["extraction_result"])
            except json.JSONDecodeError:
                row["extraction_result"] = None
        return row
