import logging
from pathlib import Path
from typing import Any, Sequence

import pytest

import neat_ml.workflow.lib_workflow as wf

def assert_logged(caplog: pytest.LogCaptureFixture, level: int, expected_message: str) -> None:
    """
    Assert a log record exists with the given level and exact message.
    """
    matched = any(rec.levelno == level and rec.getMessage() == expected_message for rec in caplog.records)
    if not matched:
        dump = "\n".join(f"[{r.levelname}] {r.getMessage()}" for r in caplog.records)
        raise AssertionError(f"Expected log at level {level} with message:\n{expected_message}\nGot:\n{dump}")

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


@pytest.mark.parametrize("ds",
    [
        {"id": "DS3", "method": "OpenCV", "detection": {}},
        {"id": "BS1", "method": "BubbleSAM", "detection": {}},
    ]
)
def test_run_detection_warns_when_paths_missing(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
    ds: dict,
) -> None:
    """
    run_detection: if 'det_dir' (or 'proc_dir') missing -> warning and return.
    """ 
    caplog.set_level(logging.WARNING) 
    ds["detection"] = {"img_dir": str(tmp_path)}
    paths = {"proc_dir": tmp_path / "p"}

    wf.run_detection(ds, paths)

    assert "Detection paths not built (step not selected or misconfig). Skipping." in caplog.text


@pytest.mark.parametrize("ds, paths",
    [
        (
            {"id": "DS4", "method": "OpenCV", "detection": {}},
            {"proc_dir": "p", "det_dir": "d"},
        ),
        (    
            {"id": "BS2", "method": "BubbleSAM", "detection": {}},
            {"det_dir": "d"},
        )
    ]
)
def test_run_detection_warns_when_img_dir_missing(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
    ds: dict,
    paths: dict,
) -> None:
    """
    run_detection: if detection.img_dir missing -> warning and return.
    """
    paths = {key: tmp_path / value for (key, value) in paths.items()}
    caplog.set_level(logging.WARNING)

    wf.run_detection(ds, paths)

    assert "No 'detection.img_dir' set for dataset" in caplog.text


@pytest.mark.parametrize("ds, paths, suff, method",
    [
        (
            {"id": "DS5", "method": "OpenCV", "detection": {}},
            {"proc_dir": "proc", "det_dir": "det"},
            "bubble_data",
            "opencv",
        ),
        (        
            {"id": "BS3", "method": "BubbleSAM", "detection": {}},
            {"det_dir": "det"},
            "masks_filtered",
            "bubblesam",
        )
    ]
)
def test_run_detection_skips_if_output_already_exists(
    caplog: pytest.LogCaptureFixture, 
    tmp_path: Path, 
    monkeypatch: pytest.MonkeyPatch,
    ds: dict,
    paths: dict,
    suff: str,
    method: str,
) -> None:
    """
    run_detection: if *_bubble_data.pkl exists -> skip and DO NOT call pipeline steps.
    """
    caplog.set_level(logging.INFO)

    proc_dir = tmp_path / "proc"
    det_dir = tmp_path / "det"
    det_dir.mkdir(parents=True)
    (det_dir / f"anything_{suff}.pkl").write_text("done")

    def bad(*_: Any, **__: Any) -> None:  # pragma: no cover
        raise AssertionError("Pipeline function should not be called when outputs exist")

    monkeypatch.setattr(wf, "cv_preprocess", bad)
    monkeypatch.setattr(wf, "collect_tiff_paths", bad)
    monkeypatch.setattr(wf, "build_df_from_img_paths", bad)
    monkeypatch.setattr(wf, f"run_{method}", bad)
    
    ds["detection"] = {"img_dir": str(tmp_path)}
    paths = {key: tmp_path / value for (key, value) in paths.items()}
    wf.run_detection(ds, paths)

    assert "Detection already exists" in caplog.text


@pytest.mark.parametrize("ds, paths, fn_calls",
    [
        (
            {"id": "BS4", "method": "bubblesam", "detection": {}},
            {"det_dir": "det"},
            ["collect", "build_df", "run"]
        ),
        (
            {"id": "DS6", "method": "OpenCV", "detection": {"debug": True}},
            {"proc_dir": "proc", "det_dir": "det"},
            ["preprocess", "collect", "build_df", "run"]
        ),
    ]
)
def test_run_detection_happy_path_calls_pipeline(
    tmp_path: Path, 
    monkeypatch: pytest.MonkeyPatch,
    ds: dict,
    paths: dict,
    fn_calls: list,
) -> None:
    """
    run_detection: happy path creates directories and calls 
    (preprocess: with ``opencv``) -> collect -> build_df -> run_opencv.
    """
    proc_dir = tmp_path / "proc"
    det_dir = tmp_path / "det"
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()

    calls: list[str] = []
    method = ds["method"].lower()
    def fake_cv_preprocess(src: Path, dst: Path) -> None:
        assert src == img_dir and dst == proc_dir
        calls.append("preprocess")

    def fake_collect(p: Path) -> list[Path]:
        assert p == proc_dir if method == "opencv" else p == img_dir
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
        if method == "opencv":
            assert debug is True
        calls.append("run")

    monkeypatch.setattr(wf, "cv_preprocess", fake_cv_preprocess)
    monkeypatch.setattr(wf, "collect_tiff_paths", fake_collect)
    monkeypatch.setattr(wf, "build_df_from_img_paths", fake_build)
    monkeypatch.setattr(wf, f"run_{method}", fake_run)
    
    ds["detection"].update({"img_dir": str(img_dir)})
    paths = {key: tmp_path / value for (key, value) in paths.items()}
    wf.run_detection(ds, paths)

    assert calls == fn_calls 
    assert det_dir.exists()
    if method == "opencv":
        assert proc_dir.exists()


@pytest.mark.parametrize("ds, paths",
    [
        (
            {"id": "DS7", "method": "OpenCV"},
            {"proc_dir": "p", "det_dir": "d"}
        ),
        (
            {"id": "BS2", "method": "BubbleSAM"},
            {"det_dir": "d"},
        )
    ]
)
def test_stage_detect_routes_to_detection_method(
    monkeypatch: pytest.MonkeyPatch, 
    tmp_path: Path,
    ds: dict,
    paths: dict,
) -> None:
    """
    stage_detect: method 'OpenCV' routes to appropriate detection method.
    """
    seen: list[str] = []

    def fake_run_detection(ds: dict[str, Any], p: dict[str, Path]) -> None:
        seen.append(ds["id"])

    monkeypatch.setattr(wf, "run_detection", fake_run_detection)

    paths = {key: tmp_path / value for (key, value) in paths.items()}
    wf.stage_detect(ds, paths)
    id_exp = ds.get("id")
    assert seen == [id_exp]


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


def test_stage_bubblesam_uses_dataset_level_img_dir_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    run_detection: with ``bubblesam`` uses dataset.img_dir when detection.img_dir is not provided.
    """
    det_dir = tmp_path / "det"
    img_dir = tmp_path / "imgs_fallback"
    img_dir.mkdir()

    used: dict[str, Path] = {}

    def fake_collect(p: Path) -> Sequence[Path]:
        used["img_dir"] = p
        return [img_dir / "im.tiff"]

    def fake_build(_: Sequence[Path]) -> dict[str, Any]:
        return {"rows": 1}

    def fake_run(_: dict[str, Any], out_dir: Path) -> None:
        used["out_dir"] = out_dir

    monkeypatch.setattr(wf, "collect_tiff_paths", fake_collect)
    monkeypatch.setattr(wf, "build_df_from_img_paths", fake_build)
    monkeypatch.setattr(wf, "run_bubblesam", fake_run)

    ds = {"id": "BS5", "method": "bubblesam", "img_dir": str(img_dir)}
    paths = {"det_dir": det_dir}

    wf.run_detection(ds, paths)

    assert used["img_dir"] == img_dir
    assert used["out_dir"] == det_dir
