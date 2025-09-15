import logging
from pathlib import Path
from typing import Any, List

import pytest

import neat_ml.workflow.lib_workflow as wf


def test_get_path_structure_builds_expected_paths(tmp_path: Path) -> None:
    """
    get_path_structure: builds proc_dir and det_dir using ds_id/method/class/time_label.
    """
    roots = {"work": str(tmp_path)}
    ds = {"id": "DS1", "method": "OpenCV", "class": "pos", "time_label": "T01"}

    paths = wf.get_path_structure(roots, ds)

    base = tmp_path / "DS1" / "OpenCV" / "pos" / "T01"
    assert paths["proc_dir"] == base / "T01_Processed_OpenCV"
    assert paths["det_dir"] == base / "T01_Processed_OpenCV_With_Blob_Data"


def test_get_path_structure_missing_work_raises_keyerror(tmp_path: Path) -> None:
    """
    If 'work' key is missing in roots, a KeyError is raised. Use match= for message compare.
    """
    roots = {"result": str(tmp_path)}
    ds = {"id": "DS1", "method": "OpenCV", "class": "pos", "time_label": "T01"}

    with pytest.raises(KeyError, match="work"):
        wf.get_path_structure(roots, ds)

def test_stage_opencv_warns_when_paths_missing(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    """
    stage_opencv: if 'det_dir' (or 'proc_dir') missing -> warning and return.
    """
    caplog.set_level(logging.WARNING)
    ds = {"id": "DS3", "method": "OpenCV", "detection": {"img_dir": str(tmp_path)}}
    paths = {"proc_dir": tmp_path / "p"}  # det_dir missing

    wf.stage_opencv(ds, paths)

    assert "Detection paths not built (step not selected or misconfig). Skipping." in caplog.text


def test_stage_opencv_warns_when_img_dir_missing(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    """
    stage_opencv: if detection.img_dir missing -> warning and return.
    """
    caplog.set_level(logging.WARNING)
    ds = {"id": "DS4", "method": "OpenCV", "detection": {}}
    paths = {"proc_dir": tmp_path / "p", "det_dir": tmp_path / "d"}

    wf.stage_opencv(ds, paths)

    assert "No 'detection.img_dir' set for dataset 'DS4'. Skipping detection." in caplog.text


def test_stage_opencv_skips_if_output_already_exists(
    caplog: pytest.LogCaptureFixture, 
    tmp_path: Path, 
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    stage_opencv: if *_bubble_data.pkl exists -> skip and DO NOT call pipeline steps.
    """
    caplog.set_level(logging.INFO)

    proc_dir = tmp_path / "proc"
    det_dir = tmp_path / "det"
    det_dir.mkdir(parents=True)
    (det_dir / "anything_bubble_data.pkl").write_text("done")

    def bad(*_: Any, **__: Any) -> None:  # pragma: no cover
        raise AssertionError("Pipeline function should not be called when outputs exist")

    monkeypatch.setattr(wf, "cv_preprocess", bad)
    monkeypatch.setattr(wf, "collect_tiff_paths", bad)
    monkeypatch.setattr(wf, "build_df_from_img_paths", bad)
    monkeypatch.setattr(wf, "run_opencv", bad)

    ds = {"id": "DS5", "method": "OpenCV", "detection": {"img_dir": str(tmp_path)}}
    paths = {"proc_dir": proc_dir, "det_dir": det_dir}

    wf.stage_opencv(ds, paths)

    assert "Detection already exists for DS5. Skipping." in caplog.text


def test_stage_opencv_happy_path_calls_pipeline(
    tmp_path: Path, 
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    stage_opencv: happy path creates directories and calls 
    preprocess -> collect -> build_df -> run_opencv.
    """
    proc_dir = tmp_path / "proc"
    det_dir = tmp_path / "det"
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()

    calls: List[str] = []

    def fake_cv_preprocess(src: Path, dst: Path) -> None:
        assert src == img_dir and dst == proc_dir
        calls.append("preprocess")

    def fake_collect(p: Path) -> list[Path]:
        assert p == proc_dir
        calls.append("collect")
        return [proc_dir / "a.tiff", proc_dir / "b.tiff"]

    def fake_build(paths: list[Path]) -> dict[str, Any]:
        assert len(paths) == 2
        calls.append("build_df")
        return {"rows": len(paths)}

    def fake_run(
        df: dict[str, Any], 
        output_dir: Path,
        *, 
        debug: bool = False
    ) -> None:
        assert df == {"rows": 2}
        assert output_dir == det_dir
        assert debug is True
        calls.append("run")

    monkeypatch.setattr(wf, "cv_preprocess", fake_cv_preprocess)
    monkeypatch.setattr(wf, "collect_tiff_paths", fake_collect)
    monkeypatch.setattr(wf, "build_df_from_img_paths", fake_build)
    monkeypatch.setattr(wf, "run_opencv", fake_run)

    ds = {
        "id": "DS6", 
        "method": "OpenCV", 
        "detection": {
            "img_dir": str(img_dir), 
            "debug": True
        }
    }
    paths = {"proc_dir": proc_dir, "det_dir": det_dir}

    wf.stage_opencv(ds, paths)

    assert calls == ["preprocess", "collect", "build_df", "run"]
    assert proc_dir.exists() and det_dir.exists()


def test_stage_detect_routes_to_opencv(
    monkeypatch: pytest.MonkeyPatch, 
    tmp_path: Path
) -> None:
    """
    stage_detect: method 'OpenCV' routes to stage_opencv.
    """
    seen: list[str] = []

    def fake_stage_opencv(ds: dict[str, Any], p: dict[str, Path]) -> None:
        seen.append(ds["id"])

    monkeypatch.setattr(wf, "stage_opencv", fake_stage_opencv)

    ds = {"id": "DS7", "method": "OpenCV"}
    paths = {"proc_dir": tmp_path / "p", "det_dir": tmp_path / "d"}

    wf.stage_detect(ds, paths)

    assert seen == ["DS7"]


def test_stage_detect_unknown_method_warns(
    caplog: pytest.LogCaptureFixture, 
    tmp_path: Path
) -> None:
    """
    stage_detect: unknown method -> warning.
    """
    caplog.set_level(logging.WARNING)
    ds = {"id": "DS8", "method": "SomethingElse"}
    paths = {"proc_dir": tmp_path / "p", "det_dir": tmp_path / "d"}

    wf.stage_detect(ds, paths)

    assert "Unknown detection method 'somethingelse' for dataset 'DS8'." in caplog.text
