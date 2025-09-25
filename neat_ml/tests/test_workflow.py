import re
import logging
from pathlib import Path
from typing import Any, List, Sequence
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
    
@pytest.mark.parametrize(
    ("steps_str", "expected"),
    [
        ("all", ["detect", "analysis"]),                # expands to full pipeline
        (" Detect ,  Analysis ", ["detect", "analysis"]),  # whitespace + case normalization
        ("ANALYSIS,DETECT", ["analysis", "detect"]),    # preserves order after lowercasing
        ("", []),                                       # empty input -> empty list
        (", ,", []),                                    # only commas/whitespace -> empty list
        ("ALL", ["all"]),                               # case-sensitive: 'ALL' does not expand
        ("detect,", ["detect"]),                        # trailing comma ignored
        ("X,DETECT", ["x", "detect"]),                  # unknown steps pass through lowercased
    ],
)
def test_as_steps_set_normalizes_and_expands(steps_str: str, expected: list[str]) -> None:
    """
    _as_steps_set: normalizes case/whitespace, preserves order, expands exact 'all',
    and passes unknown tokens through in lowercase.
    """
    assert wf._as_steps_set(steps_str) == expected

def test_get_path_structure_builds_expected_paths(tmp_path: Path) -> None:
    """
    get_path_structure: builds proc_dir and det_dir using ds_id/method/class/time_label.
    """
    roots = {"work": str(tmp_path), "results": str(tmp_path / "results")}
    ds = {"id": "DS1", "method": "OpenCV", "class": "pos", "time_label": "T01"}
    steps = ['detect','analysis']

    paths = wf.get_path_structure(roots, ds, steps)

    base = tmp_path / "DS1" / "OpenCV" / "pos" / "T01"
    assert paths["proc_dir"] == base / "T01_Processed_OpenCV"
    assert paths["det_dir"] == base / "T01_Processed_OpenCV_With_Blob_Data"

    # Default analysis outputs
    assert paths["per_csv"] == tmp_path / "results" / "DS1" / "per_image.csv"
    assert paths["agg_csv"] == tmp_path / "results" / "DS1" / "aggregate.csv"

def test_get_path_structure_missing_work_raises_keyerror(tmp_path: Path) -> None:
    """
    If 'work' key is missing in roots, a KeyError is raised. Use match= for message compare.
    """
    roots = {"result": str(tmp_path)}
    ds = {"id": "DS1", "method": "OpenCV", "class": "pos", "time_label": "T01"}
    steps = ['detect','analysis']

    with pytest.raises(KeyError, match="work"):
        wf.get_path_structure(roots, ds, steps)

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


def test_stage_bubblesam_warns_when_det_dir_missing(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    """
    stage_bubblesam: requires det_dir; if missing -> warning and return.
    """
    caplog.set_level(logging.WARNING)
    ds = {"id": "BS1", "method": "BubbleSAM", "detection": {"img_dir": str(tmp_path)}}
    paths = {"proc_dir": tmp_path / "p"}

    wf.stage_bubblesam(ds, paths)

    assert_logged(caplog, logging.WARNING, "Missing detection paths (not selected or misconfigured). Skipping.")


def test_stage_bubblesam_warns_when_img_dir_missing(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    """
    stage_bubblesam: if neither detection.img_dir nor dataset.img_dir provided -> warning and return.
    """
    caplog.set_level(logging.WARNING)
    ds = {"id": "BS2", "method": "BubbleSAM", "detection": {}}
    paths = {"det_dir": tmp_path / "d"}

    wf.stage_bubblesam(ds, paths)

    assert_logged(caplog, logging.WARNING, "No detection.img_dir set for dataset 'BS2'. Skipping.")


def test_stage_bubblesam_skips_if_output_exists(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    stage_bubblesam: if *_masks_filtered.pkl exists -> skip and DO NOT call pipeline.
    """
    caplog.set_level(logging.INFO)

    det_dir = tmp_path / "det"
    det_dir.mkdir(parents=True)
    (det_dir / "foo_masks_filtered.pkl").write_text("done")

    def bad(*_: Any, **__: Any) -> None:  # pragma: no cover
        raise AssertionError("BubbleSAM pipeline should not be called when outputs exist")

    monkeypatch.setattr(wf, "collect_tiff_paths", bad)
    monkeypatch.setattr(wf, "build_df_from_img_paths", bad)
    monkeypatch.setattr(wf, "run_bubblesam", bad)

    ds = {"id": "BS3", "method": "BubbleSAM", "detection": {"img_dir": str(tmp_path)}}
    paths = {"det_dir": det_dir}

    wf.stage_bubblesam(ds, paths)

    assert_logged(caplog, logging.INFO, "BubbleSAM outputs exist for BS3. Skipping.")


def test_stage_bubblesam_happy_path_calls_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    stage_bubblesam: happy path calls collect -> build_df -> run_bubblesam using detection.img_dir.
    """
    det_dir = tmp_path / "det"
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()

    calls: list[str] = []

    def fake_collect(p: Path) -> Sequence[Path]:
        assert p == img_dir
        calls.append("collect")
        return [img_dir / "img1.tiff", img_dir / "img2.tiff"]

    def fake_build(paths: Sequence[Path]) -> dict[str, Any]:
        assert len(paths) == 2
        calls.append("build_df")
        return {"rows": len(paths)}

    def fake_run(df: dict[str, Any], out_dir: Path) -> None:
        assert df == {"rows": 2}
        assert out_dir == det_dir
        calls.append("run_bubblesam")

    monkeypatch.setattr(wf, "collect_tiff_paths", fake_collect)
    monkeypatch.setattr(wf, "build_df_from_img_paths", fake_build)
    monkeypatch.setattr(wf, "run_bubblesam", fake_run)

    ds = {"id": "BS4", "method": "bubblesam", "detection": {"img_dir": str(img_dir)}}
    paths = {"det_dir": det_dir}

    wf.stage_bubblesam(ds, paths)

    assert calls == ["collect", "build_df", "run_bubblesam"]
    assert det_dir.exists()


def test_stage_bubblesam_uses_dataset_level_img_dir_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    stage_bubblesam: uses dataset.img_dir when detection.img_dir is not provided.
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

    wf.stage_bubblesam(ds, paths)

    assert used["img_dir"] == img_dir
    assert used["out_dir"] == det_dir

def test_stage_analyze_features_errors_when_input_dir_unavailable(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """
    stage_analyze_features: logs error when neither analysis.input_dir nor paths['det_dir'] is available.
    """
    caplog.set_level(logging.ERROR)
    ds = {"id": "AN1", "method": "OpenCV", "time_label": "T01", "analysis": {}}
    paths: dict[str, Path] = {}

    wf.stage_analyze_features(ds, paths)
    assert_logged(caplog, logging.ERROR, "No analysis input_dir provided and det_dir unavailable. Skipping 'AN1'.")


def test_stage_analyze_features_errors_when_input_dir_missing(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    """
    stage_analyze_features: logs error if input_dir path does not exist.
    """
    caplog.set_level(logging.ERROR)
    input_dir = tmp_path / "no_such_dir"
    ds = {"id": "AN2", "method": "OpenCV", "time_label": "T01", "analysis": {"input_dir": str(input_dir)}}
    paths: dict[str, Path] = {}

    wf.stage_analyze_features(ds, paths)

    assert f"Analysis input_dir '{input_dir}' does not exist for 'AN2'." in caplog.text


def test_stage_analyze_features_errors_when_composition_csv_missing(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    """
    stage_analyze_features: logs error if composition_csv is provided but does not exist.
    """
    caplog.set_level(logging.ERROR)
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    missing_csv = tmp_path / "missing.csv"

    ds = {"id": "AN3", "method": "OpenCV", "time_label": "T01", "analysis": {"input_dir": str(input_dir), "composition_csv": str(missing_csv)}}
    paths: dict[str, Path] = {}

    wf.stage_analyze_features(ds, paths)

    assert f"Composition CSV '{missing_csv}' missing for 'AN3'." in caplog.text


def test_stage_analyze_features_happy_path_calls_full_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    stage_analyze_features: happy path creates output dirs and calls full_analysis with expected args.
    """
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    (input_dir / "stub_bubble_data.pkl").write_text("stub")
    out_per = tmp_path / "out" / "per" / "per_image.csv"
    out_agg = tmp_path / "out" / "agg" / "aggregate.csv"

    called: dict[str, Any] = {}

    def fake_full_analysis(
        *,
        input_dir: Path,
        per_image_csv: Path,
        aggregate_csv: Path,
        mode: str,
        graph_method: str | None,
        graph_param: int | float | None,
        composition_csv: Path | None,
        cols_to_add: list[str],
        group_cols: list[str],
        carry_over_cols: list[str],
        time_label: str,
        exclude_numeric_cols: list[str],
    ) -> None:
        called.update(
            {
                "input_dir": input_dir,
                "per_image_csv": per_image_csv,
                "aggregate_csv": aggregate_csv,
                "mode": mode,
                "graph_method": graph_method,
                "graph_param": graph_param,
                "composition_csv": composition_csv,
                "cols_to_add": cols_to_add,
                "group_cols": group_cols,
                "carry_over_cols": carry_over_cols,
                "time_label": time_label,
                "exclude_numeric_cols": exclude_numeric_cols,
            }
        )

    monkeypatch.setattr(wf, "full_analysis", fake_full_analysis)

    ds = {
        "id": "AN4",
        "method": "OpenCV",
        "time_label": "T99",
        "composition_cols": ["PEG", "Dex"],
        "graph_method": "knn",
        "graph_param": 7,
        "analysis": {
            "input_dir": str(input_dir),
            "per_image_csv": str(out_per),
            "aggregate_csv": str(out_agg),
        },
    }
    paths: dict[str, Path] = {}

    wf.stage_analyze_features(ds, paths)

    # Output dirs created
    assert out_per.parent.exists()
    assert out_agg.parent.exists()

    # Call arguments validated
    assert called["input_dir"] == input_dir
    assert called["per_image_csv"] == out_per
    assert called["aggregate_csv"] == out_agg
    assert called["mode"] == "OpenCV"
    assert called["graph_method"] == "knn"
    assert called["graph_param"] == 7
    assert called["composition_csv"] is None
    assert called["time_label"] == "T99"
    assert called["exclude_numeric_cols"] == ["Offset"]
    assert called["cols_to_add"] == ["Group", "Phase_Separation", "PEG", "Dex"]
    assert called["carry_over_cols"] == ["Phase_Separation", "PEG", "Dex"]
    assert called["group_cols"] == ["Group", "Label", "Time", "Class"]

def _patch_logger_error_to_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Replace wf.log.error with a function that raises RuntimeError(message % args),
    so tests can assert with pytest.raises(match=...).
    """
    def _raise(msg: str, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(msg % args if args else msg)

    monkeypatch.setattr(wf.log, "error", _raise)


def test_stage_analyze_features_raises_when_input_dir_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    No analysis.input_dir and no paths['det_dir'] -> logs (now raises) and returns.
    """
    _patch_logger_error_to_raise(monkeypatch)

    ds = {"id": "AN1", "method": "OpenCV", "time_label": "T01", "analysis": {}}
    with pytest.raises(
        RuntimeError,
        match=r"No analysis input_dir provided and det_dir unavailable\. Skipping 'AN1'\.",
    ):
        wf.stage_analyze_features(ds, paths={})


def test_stage_analyze_features_raises_when_input_dir_path_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    analysis.input_dir exists as a string but the path itself does not exist.
    """
    _patch_logger_error_to_raise(monkeypatch)

    missing_dir = tmp_path / "no_such_dir"
    ds = {
        "id": "AN2",
        "method": "OpenCV",
        "time_label": "T01",
        "analysis": {"input_dir": str(missing_dir)},
    }
    pattern = rf"Analysis input_dir '{re.escape(str(missing_dir))}' does not exist for 'AN2'\."
    with pytest.raises(RuntimeError, match=pattern):
        wf.stage_analyze_features(ds, paths={})


def test_stage_analyze_features_raises_when_composition_csv_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    composition_csv provided but the file does not exist.
    """
    _patch_logger_error_to_raise(monkeypatch)

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    missing_csv = tmp_path / "missing.csv"

    ds = {
        "id": "AN3",
        "method": "OpenCV",
        "time_label": "T01",
        "analysis": {"input_dir": str(input_dir), "composition_csv": str(missing_csv)},
    }
    pattern = rf"Composition CSV '{re.escape(str(missing_csv))}' missing for 'AN3'\."
    with pytest.raises(RuntimeError, match=pattern):
        wf.stage_analyze_features(ds, paths={})


def test_stage_analyze_features_raises_when_no_detection_outputs_opencv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Input dir exists but contains no *_bubble_data.pkl files (mode='OpenCV').
    """
    _patch_logger_error_to_raise(monkeypatch)

    input_dir = tmp_path / "empty_in"
    input_dir.mkdir()

    ds = {
        "id": "AN4",
        "method": "OpenCV",
        "time_label": "T01",
        "analysis": {"input_dir": str(input_dir)},
    }
    pat = re.escape("*_bubble_data.pkl")
    expected = (
        rf"No detection outputs matching '{pat}' under '{re.escape(str(input_dir))}' "
        rf"for dataset 'AN4' \(mode='OpenCV'\)\. Skipping\."
    )
    with pytest.raises(RuntimeError, match=expected):
        wf.stage_analyze_features(ds, paths={})


def test_stage_analyze_features_raises_when_no_detection_outputs_bubblesam(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Input dir exists but contains no *_masks_filtered.pkl files (mode='BubbleSAM').
    """
    _patch_logger_error_to_raise(monkeypatch)

    input_dir = tmp_path / "empty_in_bs"
    input_dir.mkdir()

    ds = {
        "id": "AN5",
        "method": "BubbleSAM",
        "time_label": "T01",
        "analysis": {"input_dir": str(input_dir)},
    }
    pat = re.escape("*_masks_filtered.pkl")
    expected = (
        rf"No detection outputs matching '{pat}' under '{re.escape(str(input_dir))}' "
        rf"for dataset 'AN5' \(mode='BubbleSAM'\)\. Skipping\."
    )
    with pytest.raises(RuntimeError, match=expected):
        wf.stage_analyze_features(ds, paths={})

def test_stage_analyze_features_logs_when_input_dir_falsy_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EmptyStrPath:
        def __init__(self, value: str = "") -> None:
            self._v = value
        def __str__(self) -> str:
            return ""
        def exists(self) -> bool:  # pragma: no cover
            return False
        def rglob(self, pattern: str):  # pragma: no cover
            return iter(())
        @property
        def parent(self):  # pragma: no cover
            return self
        def mkdir(self, parents: bool = False, exist_ok: bool = False) -> None:  # pragma: no cover
            return None
    monkeypatch.setattr(wf, "Path", EmptyStrPath)
    def _raise(msg: str, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(msg % args if args else msg)
    monkeypatch.setattr(wf.log, "error", _raise)

    ds = {"id": "AN1", "method": "OpenCV", "time_label": "T01", "analysis": {}}
    with pytest.raises(
        RuntimeError,
        match=r"No analysis input_dir provided and det_dir unavailable\. Skipping 'AN1'\.",
    ):
        wf.stage_analyze_features(ds, paths={})
