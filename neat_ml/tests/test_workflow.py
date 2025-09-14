import re
import logging
from pathlib import Path
from typing import Any, List, Sequence, Dict

import pandas as pd
import pytest

import neat_ml.workflow.lib_workflow as wf

def assert_logged(caplog: pytest.LogCaptureFixture, level: int, expected_message: str) -> None:
    """
    Assert a log record exists with the given level and exact message.
    """
    matched = any(rec.levelno == level and rec.getMessage() == expected_message for rec in caplog.records)
    if not matched:
        dump = "\n".join(f"[{r.levelname}] {r.getMessage()}" for r in caplog.records)
        raise AssertionError(
            f"Expected log at level {level} with message:\n{expected_message}\nGot:\n{dump}"
        )

def _patch_logger_error_to_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Replace wf.log.error with a function that raises RuntimeError(message % args),
    so tests can assert with pytest.raises(match=...).
    """
    def _raise(msg: str, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(msg % args if args else msg)

    monkeypatch.setattr(wf.log, "error", _raise)

@pytest.mark.parametrize(
    ("steps_str", "expected"),
    [
        # 'all' expands to full pipeline in the new implementation
        ("all", ["detect", "analysis", "train", "infer", "explain", "plot"]),
        (" Detect ,  Analysis ", ["detect", "analysis"]),  # whitespace + case normalization
        ("ANALYSIS,DETECT", ["analysis", "detect"]),       # preserves order after lowercasing
        ("", []),                                          # empty input -> empty list
        (", ,", []),                                       # only commas/whitespace -> empty list
        ("ALL", ["all"]),                                  # case-sensitive: 'ALL' does not expand
        ("detect,", ["detect"]),                           # trailing comma ignored
        ("X,DETECT", ["x", "detect"]),                     # unknown steps pass through lowercased
    ],
)
def test_as_steps_set_normalizes_and_expands(steps_str: str, expected: list[str]) -> None:
    assert wf._as_steps_set(steps_str) == expected

def test_get_path_structure_builds_expected_paths(tmp_path: Path) -> None:
    """
    get_path_structure: builds proc_dir and det_dir using ds_id/method/class/time_label.
    """
    roots = {"work": str(tmp_path), "results": str(tmp_path / "results")}
    ds = {"id": "DS1", "method": "OpenCV", "class": "pos", "time_label": "T01"}
    steps = ["detect", "analysis"]

    paths = wf.get_path_structure(roots, ds, steps)

    base = tmp_path / "DS1" / "OpenCV" / "pos" / "T01"
    assert paths["proc_dir"] == base / "T01_Processed_OpenCV"
    assert paths["det_dir"] == base / "T01_Processed_OpenCV_With_Blob_Data"

    assert paths["per_csv"] == tmp_path / "results" / "DS1" / "per_image.csv"
    assert paths["agg_csv"] == tmp_path / "results" / "DS1" / "aggregate.csv"

def test_get_path_structure_missing_work_raises_keyerror(tmp_path: Path) -> None:
    """
    If 'work' key is missing in roots, a KeyError is raised. Use match= for message compare.
    """
    roots = {"result": str(tmp_path)}
    ds = {"id": "DS1", "method": "OpenCV", "class": "pos", "time_label": "T01"}
    steps = ["detect", "analysis"]

    with pytest.raises(KeyError, match="work"):
        wf.get_path_structure(roots, ds, steps)


def test_get_path_structure_includes_train_infer_explain_and_model_override(tmp_path: Path) -> None:
    """
    When steps include train/infer/explain/plot, ensure the extra paths are built and
    model_dir honors roots['model'] override.
    """
    roots = {
        "work": str(tmp_path),
        "results": str(tmp_path / "results"),
        "model": str(tmp_path / "custom_model_dir"),
    }
    ds = {"id": "DS2", "method": "BubbleSAM", "class": "neg", "time_label": "T02", "composition_csv": "comp.csv"}
    steps = ["analysis", "train", "infer", "explain", "plot"]

    paths = wf.get_path_structure(roots, ds, steps)

    results_root = tmp_path / "results"
    assert paths["per_csv"] == results_root / "DS2" / "per_image.csv"
    assert paths["agg_csv"] == results_root / "DS2" / "aggregate.csv"
    assert paths["composition_csv"] == Path("comp.csv")
    assert paths["model_dir"] == tmp_path / "custom_model_dir"
    assert paths["explain_dir"] == results_root / "DS2" / "explain"
    assert paths["pred_csv"] == results_root / "infer_DS2" / "pred.csv"
    assert paths["phase_dir"] == results_root / "infer_DS2" / "phase_plots"

def test_stage_opencv_warns_when_paths_missing(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    """
    stage_opencv: if 'det_dir' (or 'proc_dir') missing -> warning and return.
    """
    caplog.set_level(logging.WARNING)
    ds = {"id": "DS3", "method": "OpenCV", "detection": {"img_dir": str(tmp_path)}}
    paths = {"proc_dir": tmp_path / "p"}  # det_dir missing

    wf.stage_opencv(ds, paths)

    assert_logged(caplog, logging.WARNING, "Detection paths not built (step not selected or misconfig). Skipping.")


def test_stage_opencv_warns_when_img_dir_missing(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    """
    stage_opencv: if detection.img_dir missing -> warning and return.
    """
    caplog.set_level(logging.WARNING)
    ds = {"id": "DS4", "method": "OpenCV", "detection": {}}
    paths = {"proc_dir": tmp_path / "p", "det_dir": tmp_path / "d"}

    wf.stage_opencv(ds, paths)

    assert_logged(caplog, logging.WARNING, "No 'detection.img_dir' set for dataset 'DS4'. Skipping detection.")


def test_stage_opencv_skips_if_output_already_exists(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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

    assert_logged(caplog, logging.INFO, "Detection already exists for DS5. Skipping.")


def test_stage_opencv_happy_path_calls_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
        debug: bool = False,
    ) -> None:
        assert df == {"rows": 2}
        assert output_dir == det_dir
        assert debug is True
        calls.append("run")

    monkeypatch.setattr(wf, "cv_preprocess", fake_cv_preprocess)
    monkeypatch.setattr(wf, "collect_tiff_paths", fake_collect)
    monkeypatch.setattr(wf, "build_df_from_img_paths", fake_build)
    monkeypatch.setattr(wf, "run_opencv", fake_run)

    ds = {"id": "DS6", "method": "OpenCV", "detection": {"img_dir": str(img_dir), "debug": True}}
    paths = {"proc_dir": proc_dir, "det_dir": det_dir}

    wf.stage_opencv(ds, paths)

    assert calls == ["preprocess", "collect", "build_df", "run"]
    assert proc_dir.exists() and det_dir.exists()

def test_stage_detect_routes_to_opencv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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


def test_stage_detect_routes_to_bubblesam(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    stage_detect: method 'BubbleSAM' routes to stage_bubblesam.
    """
    seen: list[str] = []

    def fake_stage_bubblesam(ds: dict[str, Any], p: dict[str, Path]) -> None:
        seen.append(ds["id"])

    monkeypatch.setattr(wf, "stage_bubblesam", fake_stage_bubblesam)

    ds = {"id": "DS7B", "method": "BubbleSAM"}
    paths = {"det_dir": tmp_path / "d"}

    wf.stage_detect(ds, paths)

    assert seen == ["DS7B"]


def test_stage_detect_unknown_method_warns(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    """
    stage_detect: unknown method -> warning.
    """
    caplog.set_level(logging.WARNING)
    ds = {"id": "DS8", "method": "SomethingElse"}
    paths = {"proc_dir": tmp_path / "p", "det_dir": tmp_path / "d"}

    wf.stage_detect(ds, paths)

    assert_logged(caplog, logging.WARNING, "Unknown detection method 'somethingelse' for dataset 'DS8'.")

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


def test_stage_analyze_features_errors_when_input_dir_missing(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """
    stage_analyze_features: logs error if input_dir path does not exist.
    """
    caplog.set_level(logging.ERROR)
    input_dir = tmp_path / "no_such_dir"
    ds = {"id": "AN2", "method": "OpenCV", "time_label": "T01", "analysis": {"input_dir": str(input_dir)}}
    paths: dict[str, Path] = {}

    wf.stage_analyze_features(ds, paths)

    assert_logged(caplog, logging.ERROR, f"Analysis input_dir '{input_dir}' does not exist for 'AN2'.")


def test_stage_analyze_features_errors_when_composition_csv_missing(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """
    stage_analyze_features: logs error if composition_csv is provided but does not exist.
    """
    caplog.set_level(logging.ERROR)
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    missing_csv = tmp_path / "missing.csv"

    ds = {
        "id": "AN3",
        "method": "OpenCV",
        "time_label": "T01",
        "analysis": {"input_dir": str(input_dir), "composition_csv": str(missing_csv)},
    }
    paths: dict[str, Path] = {}

    wf.stage_analyze_features(ds, paths)

    assert_logged(caplog, logging.ERROR, f"Composition CSV '{missing_csv}' missing for 'AN3'.")


def test_stage_analyze_features_happy_path_calls_full_analysis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_stage_analyze_features_raises_when_input_dir_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_stage_analyze_features_logs_when_input_dir_falsy_string(monkeypatch: pytest.MonkeyPatch) -> None:
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

def test_stage_train_model_requires_validation_args(tmp_path: Path) -> None:
    train_ds = {"id": "TR1"}
    train_paths = {"agg_csv": tmp_path / "train.csv", "model_dir": tmp_path / "model"}

    with pytest.raises(ValueError, match=r"requires a validation dataset config \(val_ds\)\."):
        wf.stage_train_model(train_ds, train_paths, val_ds=None, val_paths=None)

    with pytest.raises(ValueError, match=r"requires validation paths \(val_paths\)\."):
        wf.stage_train_model(train_ds, train_paths, val_ds={"id": "VAL"}, val_paths=None)


def test_stage_train_model_missing_train_csv_raises(tmp_path: Path) -> None:
    train_ds = {"id": "TR2"}
    missing = (tmp_path / "train.csv").resolve()
    train_paths = {"agg_csv": missing, "model_dir": tmp_path / "model"}
    val_paths = {"agg_csv": tmp_path / "val.csv"}
    val_paths["agg_csv"].write_text("Phase_Separation\n0\n1\n")

    with pytest.raises(FileNotFoundError, match=re.escape(f"Train aggregate CSV not found: {missing}")):
        wf.stage_train_model(train_ds, train_paths, val_ds={"id": "VAL"}, val_paths=val_paths)

def test_stage_train_model_missing_val_csv_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    train_ds = {"id": "TR3"}
    train_paths = {"agg_csv": tmp_path / "train.csv", "model_dir": tmp_path / "model"}
    train_paths["agg_csv"].write_text("Phase_Separation\n0\n1\n")

    missing = tmp_path / "val.csv"
    monkeypatch.setattr(
        wf, "ml_preprocess",
        lambda df, target, exclude: (pd.DataFrame({"f": [0, 1]}), pd.Series([0, 1], name=target))
    )

    with pytest.raises(FileNotFoundError, match=re.escape(f"Validation aggregate CSV not found: {missing}")):
        wf.stage_train_model(train_ds, train_paths, val_ds={"id": "VAL"}, val_paths={"agg_csv": missing})


def test_stage_train_model_no_overlapping_features_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    train_ds = {"id": "TR4"}
    train_paths = {"agg_csv": tmp_path / "train.csv", "model_dir": tmp_path / "model"}
    val_paths = {"agg_csv": tmp_path / "val.csv"}
    train_paths["agg_csv"].write_text("Phase_Separation\n0\n1\n")
    val_paths["agg_csv"].write_text("Phase_Separation\n1\n0\n")

    # First call (train) -> columns a,b ; second call (val) -> columns c,d
    state = {"i": 0}

    def fake_ml_preprocess(df: pd.DataFrame, target: str, exclude: list[str]):
        if state["i"] == 0:
            state["i"] += 1
            X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
            y = pd.Series([0, 1], name=target)
            return X, y
        else:
            X = pd.DataFrame({"c": [5, 6], "d": [7, 8]})
            y = pd.Series([1, 0], name=target)
            return X, y

    monkeypatch.setattr(wf, "ml_preprocess", fake_ml_preprocess)

    with pytest.raises(ValueError, match="No overlapping feature columns between train and validation."):
        wf.stage_train_model(train_ds, train_paths, val_ds={"id": "VAL"}, val_paths=val_paths)


def test_stage_train_model_happy_path_saves_bundle_and_roc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)

    train_ds = {"id": "TR5"}
    train_paths = {"agg_csv": tmp_path / "train.csv", "model_dir": tmp_path / "model"}
    val_paths = {"agg_csv": tmp_path / "val.csv"}
    train_paths["agg_csv"].write_text("Phase_Separation\n0\n1\n0\n")
    val_paths["agg_csv"].write_text("Phase_Separation\n1\n0\n1\n")

    # Train has a,b,c ; Val has b,c,d -> common b,c (mismatch warning expected)
    call_idx = {"i": 0}

    def fake_ml_preprocess(df: pd.DataFrame, target: str, exclude: list[str]):
        if call_idx["i"] == 0:
            call_idx["i"] += 1
            X = pd.DataFrame({"a": [1, 2, 3], "b": [0, 1, 0], "c": [0.2, 0.3, 0.4]})
            y = pd.Series([0, 1, 0], name=target)
            return X, y
        else:
            X = pd.DataFrame({"b": [1, 0, 1], "c": [0.5, 0.6, 0.7], "d": [2, 3, 4]})
            y = pd.Series([1, 0, 1], name=target)
            return X, y

    monkeypatch.setattr(wf, "ml_preprocess", fake_ml_preprocess)

    seen: Dict[str, Any] = {"save": None, "roc": None}

    def fake_train_with_validation(X_tr, y_tr, X_val, y_val):
        # Ensure alignment to common features
        assert list(X_tr.columns) == ["b", "c"]
        assert list(X_val.columns) == ["b", "c"]
        metrics = {"val_roc_auc": 0.88, "val_pr_auc": 0.77}
        best = {"param": 1}
        proba = [0.1, 0.9, 0.8]
        return object(), metrics, best, proba

    def fake_save_model_bundle(model, features, metrics, best_params, path: str):
        seen["save"] = {"features": list(features), "metrics": metrics, "best_params": best_params, "path": path}

    def fake_plot_roc(y_true, y_prob, out_png: str):
        seen["roc"] = {"y_true": list(y_true), "y_prob": list(y_prob), "out_png": out_png}

    monkeypatch.setattr(wf, "train_with_validation", fake_train_with_validation)
    monkeypatch.setattr(wf, "save_model_bundle", fake_save_model_bundle)
    monkeypatch.setattr(wf, "plot_roc", fake_plot_roc)

    model_path = wf.stage_train_model(train_ds, train_paths, val_ds={"id": "VAL"}, val_paths=val_paths)

    assert_logged(caplog, logging.WARNING, "Feature mismatch: using 2 common features (train=3, val=3).")

    expected_model_path = train_paths["model_dir"] / "TR5_model.joblib"
    assert model_path == expected_model_path
    assert train_paths["model_dir"].exists()

    assert seen["save"]["features"] == ["b", "c"]
    assert seen["save"]["path"] == str(expected_model_path)

    assert seen["roc"]["y_true"] == [1, 0, 1]
    assert seen["roc"]["y_prob"] == [0.1, 0.9, 0.8]
    assert Path(seen["roc"]["out_png"]).name == "TR5_val_roc.png"

def test_stage_explain_aligns_features_and_calls_compare_methods(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agg_csv = tmp_path / "train_agg.csv"
    agg_csv.write_text("Phase_Separation\n0\n1\n1\n")

    def fake_ml_preprocess(df: pd.DataFrame, target: str, exclude: list[str]):
        X = pd.DataFrame({"a": [1, 2, 3], "b": [0, 1, 0], "c": [0.2, 0.3, 0.4]})
        y = pd.Series([0, 1, 1], name=target)
        return X, y

    monkeypatch.setattr(wf, "ml_preprocess", fake_ml_preprocess)

    def fake_joblib_load(path: Path):
        return {"model": object(), "features": ["b", "c"]}

    monkeypatch.setattr(wf, "joblib_load", fake_joblib_load)

    captured: Dict[str, Any] = {}

    def fake_compare_methods(model, X: pd.DataFrame, y: pd.Series, out_dir: Path, top: int):
        captured["cols"] = list(X.columns)
        captured["y"] = list(y)
        captured["out_dir"] = out_dir
        captured["top"] = top

    monkeypatch.setattr(wf, "compare_methods", fake_compare_methods)

    train_ds = {"id": "TRX", "composition_cols": ["PEG"]}
    paths = {"agg_csv": agg_csv, "explain_dir": tmp_path / "explain_out"}
    model_path = tmp_path / "bundle.joblib"

    wf.stage_explain(train_ds, paths, model_path)

    assert captured["cols"] == ["b", "c"]
    assert captured["y"] == [0, 1, 1]
    assert captured["out_dir"] == paths["explain_dir"]
    assert captured["top"] == 20

def test_stage_run_inference_calls_inference_and_makes_pred_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agg_csv = tmp_path / "agg.csv"
    agg_csv.write_text("Phase_Separation\n0\n1\n")

    called: Dict[str, Any] = {}

    def fake_run_inference(model_in: Path, data_csv: Path, target: str, exclude_cols: list[str], pred_csv: Path):
        called.update(
            {
                "model_in": model_in,
                "data_csv": data_csv,
                "target": target,
                "exclude_cols": exclude_cols,
                "pred_csv": pred_csv,
            }
        )
        pred_csv.write_text("Phase_Separation,Pred_Label\n0,0\n1,1\n")

    monkeypatch.setattr(wf, "run_inference", fake_run_inference)

    ds = {"id": "INFER1", "composition_cols": ["Dex", "PEG"]}
    paths = {
        "agg_csv": agg_csv,
        "pred_csv": tmp_path / "infer" / "pred.csv",
        "phase_dir": tmp_path / "infer" / "phase_plots",
    }
    model_path = tmp_path / "model.joblib"

    wf.stage_run_inference_and_plot(ds, paths, model_path, steps=["infer"])

    assert called["model_in"] == model_path
    assert called["data_csv"] == agg_csv
    assert called["target"] == "Phase_Separation"
    assert called["pred_csv"] == paths["pred_csv"]
    assert called["exclude_cols"] == ["Group", "Label", "Time", "Class", "Offset", "Dex", "PEG"]
    assert paths["pred_csv"].parent.exists()


def test_stage_run_inference_and_plot_skips_plot_when_wrong_num_composition_cols(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)

    pred_csv = tmp_path / "pred.csv"
    pred_csv.write_text("Phase_Separation,Pred_Label\n0,0\n1,1\n")

    ds = {"id": "INFER2", "composition_cols": ["A", "B", "C"]}  # 3 columns -> skip plot
    paths = {
        "agg_csv": tmp_path / "agg.csv",  # not used here
        "pred_csv": pred_csv,
        "phase_dir": tmp_path / "phase",
    }

    wf.stage_run_inference_and_plot(ds, paths, model_path=tmp_path / "m.joblib", steps=["plot"])

    assert_logged(caplog, logging.WARNING, "Skipping plot for INFER2: requires 2 composition columns.")


def test_stage_run_inference_and_plot_plot_only_missing_pred_csv_raises(tmp_path: Path) -> None:
    ds = {"id": "INFER3", "composition_cols": ["Dex", "PEG"]}
    paths = {
        "agg_csv": tmp_path / "agg.csv",
        "pred_csv": tmp_path / "does_not_exist.csv",
        "phase_dir": tmp_path / "phase",
    }

    with pytest.raises(FileNotFoundError, match=re.escape(str(paths["pred_csv"]))):
        wf.stage_run_inference_and_plot(ds, paths, model_path=tmp_path / "m.joblib", steps=["plot"])


def test_stage_run_inference_and_plot_calls_construct_phase_diagram(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pred_csv = tmp_path / "pred.csv"
    pred_csv.write_text(
        "Dex,PEG,Phase_Separation,Pred_Label\n"
        "10,90,1,1\n"
        "50,50,0,0\n"
        "70,30,1,1\n"
    )

    captured: Dict[str, Any] = {}

    def fake_construct_phase_diagram(
        df: pd.DataFrame,
        dex_col: str,
        peo_col: str,
        true_phase_col: str,
        pred_phase_col: str,
        out_dir: Path,
        title: str,
        fname: str,
    ):
        captured.update(
            {
                "cols": list(df.columns),
                "dex_col": dex_col,
                "peo_col": peo_col,
                "true_phase_col": true_phase_col,
                "pred_phase_col": pred_phase_col,
                "out_dir": out_dir,
                "title": title,
                "fname": fname,
            }
        )

    monkeypatch.setattr(wf, "construct_phase_diagram", fake_construct_phase_diagram)

    ds = {"id": "INFER4", "composition_cols": ["Dex", "PEG"]}
    paths = {"pred_csv": pred_csv, "phase_dir": tmp_path / "phase", "agg_csv": tmp_path / "agg.csv"}

    wf.stage_run_inference_and_plot(ds, paths, model_path=tmp_path / "m.joblib", steps=["plot"])

    assert captured["dex_col"] == "Dex"
    assert captured["peo_col"] == "PEG"
    assert captured["true_phase_col"] == "Phase_Separation"
    assert captured["pred_phase_col"] == "Pred_Label"
    assert captured["out_dir"] == paths["phase_dir"]
    assert captured["title"] == "INFER4"
    assert captured["fname"] == "phase_diagram"
    assert captured["cols"] == ["Dex", "PEG", "Phase_Separation", "Pred_Label"]
