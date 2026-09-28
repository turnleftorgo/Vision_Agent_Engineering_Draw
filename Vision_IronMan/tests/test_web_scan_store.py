from __future__ import annotations

from pathlib import Path

from web_backend.store import ScanStore


def refined_event(root: Path, *, key: str, fai: str) -> dict:
    page = root / "page.png"
    crop = root / f"{key}.png"
    page.touch()
    crop.touch()
    return {
        "page_index": 2,
        "candidate_key": key,
        "fai_number": fai,
        "spc_code": "FC",
        "description": "inner width",
        "page_width": 2000,
        "page_height": 1000,
        "final_bbox_full_image": [100, 200, 500, 600],
        "marker_bbox_full_image": [110, 210, 160, 260],
        "refined_crop_path_absolute": str(crop),
        "page_image": str(page),
    }


def test_future_extraction_fields_remain_nullable_and_fai_is_sorted(tmp_path: Path) -> None:
    store = ScanStore(tmp_path / "scan.sqlite3")
    store.create_run("run", "drawing.pdf", tmp_path / "drawing.pdf", tmp_path)
    store.upsert_refined_record("run", refined_event(tmp_path, key="b", fai="1173"))
    store.upsert_refined_record("run", refined_event(tmp_path, key="a", fai="184"))

    records = store.list_records("run")
    assert [item["fai"] for item in records] == ["184", "1173"]
    assert records[0]["project"] is None
    assert records[0]["revision"] is None
    assert records[0]["nominal"] is None
    assert records[0]["points"] is None


def test_user_crop_overrides_model_without_deleting_it(tmp_path: Path) -> None:
    store = ScanStore(tmp_path / "scan.sqlite3")
    store.create_run("run", "drawing.pdf", tmp_path / "drawing.pdf", tmp_path)
    record_id = store.upsert_refined_record(
        "run", refined_event(tmp_path, key="candidate", fai="449")
    )
    user_crop = tmp_path / "user.png"
    user_crop.touch()
    store.save_user_crop(record_id, [80, 160, 620, 700], user_crop)

    record = store.get_record("run", record_id)
    assert record is not None
    assert record["model_bbox"] == [100, 200, 500, 600]
    assert record["effective_bbox"] == [80, 160, 620, 700]
    assert record["effective_crop_path"] == str(user_crop)
    assert record["user_override"] is True


def test_progress_uses_discovered_fai_candidates_as_denominator(tmp_path: Path) -> None:
    store = ScanStore(tmp_path / "scan.sqlite3")
    store.create_run("run", "drawing.pdf", tmp_path / "drawing.pdf", tmp_path)
    store.save_stage1_progress(
        "run",
        {
            "page_index": 1,
            "page_width": 2000,
            "page_height": 1000,
            "visualization_path": str(tmp_path / "overview.jpg"),
            "modules": [
                {"bbox_pixels": [0, 0, 1000, 500]},
                {"bbox_pixels": [1000, 0, 2000, 500]},
            ],
        },
    )
    store.save_stage2_progress(
        "run",
        {
            "page_index": 1,
            "page_width": 2000,
            "page_height": 1000,
            "module_index": 1,
            "module_bbox_full_image": [0, 0, 1000, 500],
            "module_width": 1000,
            "module_height": 500,
            "valid_cluster_count": 4,
            "valid_clusters": [],
            "visualization_path": str(tmp_path / "module.jpg"),
        },
    )
    store.upsert_refined_record("run", refined_event(tmp_path, key="a", fai="184"))
    store.upsert_refined_record("run", refined_event(tmp_path, key="b", fai="449"))

    progress = store.get_progress("run")
    assert progress["stage1_module_count"] == 2
    assert progress["candidate_count"] == 4
    assert progress["refined_count"] == 2
    assert progress["percent"] == 50
