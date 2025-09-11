import re
import logging
from pathlib import Path
import pytest
import os
import shutil
import torch
import pandas as pd
from numpy.testing import assert_allclose

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

def test_get_path_structure_builds_expected_paths(tmp_path: Path):
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

def test_get_path_structure_missing_work_raises_keyerror(tmp_path: Path):
    """
    If 'work' key is missing in roots, a KeyError is raised.
    """
    roots = {"result": str(tmp_path)}
    ds = {"id": "DS1", "method": "OpenCV", "class": "pos", "time_label": "T01"}
    steps = ['detect','analysis']

    with pytest.raises(KeyError, match="work"):
        wf.get_path_structure(roots, ds, steps)

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
):
    """
    run_detection: if 'det_dir' (or 'proc_dir') missing -> warning and return.
    """ 
    caplog.set_level(logging.WARNING) 
    ds["detection"] = {"img_dir": tmp_path}
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
):
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
    ds: dict,
    paths: dict,
    suff: str,
    method: str,
):
    """
    stage_opencv: if *_bubble_data.parquet.gzip exists -> skip and DO NOT call pipeline steps.
    """
    caplog.set_level(logging.INFO)

    det_dir = tmp_path / "det"
    det_dir.mkdir(parents=True)
    (det_dir / f"anything_{suff}.parquet.gzip").write_text("done")
    
    ds["detection"] = {"img_dir": tmp_path}
    paths = {key: tmp_path / value for (key, value) in paths.items()}
    wf.run_detection(ds, paths)

    assert "Detection already exists" in caplog.text


@pytest.mark.parametrize("ds, paths, exp_columns",
    [
        (    
            {"id": "BS4", "method": "bubblesam", "detection": {}},
            {"det_dir": "det_dir"},
            {"image_filepath", "num_blobs_SAM", "median_radii_SAM"},
        ),
        (
            {"id": "DS6", "method": "OpenCV", "detection": {"debug": True}},
            {"proc_dir": "proc_dir", "det_dir": "det_dir"},
            {"image_filepath", "num_blobs_opencv", "median_radii_opencv"},
        ),
    ]
)
def test_stage_detect_pipeline_runs(
    tmpdir,
    mask_settings,
    reference_images: tuple,
    ds: dict,
    paths: dict,
    exp_columns: set,
):
    """
    test that ``stage_detect`` runs successfully
    when provided the appropriate directories.
    """
    raw_image = Path(reference_images[3])
    method = ds.get("method")
    with tmpdir.as_cwd():
        proc_dir = Path("proc_dir")
        det_dir = Path("det_dir")
        img_dir = Path("img_dir")
        tif_dir = img_dir / "tif_img"
        tiff_dir = img_dir / "tiff_img"
        tif_dir.mkdir(parents=True)
        tiff_dir.mkdir()
        # make copies of ``raw_image`` in ``img_dir``
        # with the extensions ``.tiff`` and ``.tif``
        shutil.copy(raw_image, tif_dir / "tif_img.tif")
        shutil.copy(raw_image, tiff_dir / "tiff_img.tiff")
        
        if method == "bubblesam":
            # provide reduced mask_settings/model params
            ds["detection"].update(
                {"model_cfg":
                    {
                        "mask_settings": mask_settings,
                        "checkpoint_path": "facebook/sam2.1-hiera-tiny",
                        "device": "gpu",
                    }
                }
            )

        ds["detection"]["img_dir"] = img_dir
        paths = {
            key: Path(tmpdir) / value for key, value in paths.items()
        } 
        df_out = wf.stage_detect(ds, paths)
        
        proc_dir_exp = Path(tmpdir) / proc_dir
        det_dir_exp = Path(tmpdir) / det_dir
    # assert that the ``.tiff`` and ``.tif`` images in subdirectores
    # of ``img_dir`` were preprocessed and that detection was run.
    if method == "OpenCV":
        assert (set(os.listdir(proc_dir_exp)) ==
            set(['tif_img.tif', 'tiff_img.tiff']))
        assert (set(os.listdir(det_dir_exp)) == 
            set(
                [
                    '.joblib_cache',
                    'tiff_img_bubble_data.parquet.gzip',
                    'tiff_img_debug.png'
                ]
            )
        )
    if method == "bubblesam":
        assert (set(os.listdir(det_dir_exp)) == 
            set(
                [
                    'tiff_img_masks_filtered.parquet.gzip',
                    'bubblesam_summary.csv',
                ]
            )
        )
    
    # here and above, check that the absolute file paths for ``proc_dir``
    # and ``det_dir`` are generated by ``stage_opencv``
    assert proc_dir_exp.is_absolute() 
    assert det_dir_exp.is_absolute()
    assert df_out.shape == (1, 3)
    assert exp_columns.issubset(df_out.columns)


def test_stage_detect_unknown_method_error(
    tmp_path: Path
):
    """
    stage_detect: unknown method -> ValueError.
    """
    ds = {"id": "DS8", "method": "SomethingElse"}
    paths = {"proc_dir": tmp_path / "p", "det_dir": tmp_path / "d"}
    
    with pytest.raises(ValueError, match="Unknown detection method"):
        wf.stage_detect(ds, paths)


@pytest.mark.parametrize("method, device, paths, suffix",
    [
        # cases using bubblesam method with different devices
        ("bubblesam", "cpu", {"det_dir": "det"}, "masks_filtered"),
        pytest.param(
            "bubblesam",
            "gpu",
            {"det_dir": "det"},
            "masks_filtered",
            marks=[
                pytest.mark.skipif(
                    not torch.backends.mps.is_available()
                    and torch.cuda.is_available(), reason="only run when gpu available"
                )
            ]
        ),
        # case using opencv method
        ("opencv", None, {"det_dir": "det", "proc_dir": "proc"}, "bubble_data")
    ]
)
def test_run_workflow_single_image_path(
    tmp_path,
    image_with_circles_fixture,
    mask_settings,
    method,
    device,
    paths,
    suffix,
):
    """
    test that providing run workflow with a path to a single image
    generates the correct outputs
    """
    det_dir = tmp_path / paths.get("det_dir")

    ds = {
        "id": "BS5",
        "method": method,
        "detection": {
            "img_dir": image_with_circles_fixture,
        }
    }
    if method == "bubblesam":
        ds["detection"].update(
            {"model_cfg":
                {
                    "mask_settings": mask_settings,
                    "checkpoint_path": "facebook/sam2.1-hiera-tiny",
                    "device": device,
                }
            }
        )
    paths = {key: tmp_path / value for (key, value) in paths.items()}

    df_out = wf.run_detection(ds, paths)
    assert df_out.shape == (1, 3)
    masks = pd.read_parquet(det_dir / f"circles_{suffix}.parquet.gzip")
    if method == "bubblesam":
        summary_df = pd.read_csv(det_dir / "bubblesam_summary.csv")
        assert_allclose(summary_df.median_radii_SAM, 12.778613837669742)
        assert summary_df.num_blobs_SAM.item() == 2
        assert_allclose(masks.major_axis, [30.05063897290423, 20.101948365163757])
        assert_allclose(masks.radius, [15.022706457370044, 10.045109950630787])
        assert_allclose(masks.area, [709.0, 317.0])
        assert all(masks.circ) > 0.90
    else:
        assert_allclose(masks.radius, [14.560219764709473, 9.486832618713379])
        assert_allclose(masks.area, [666.017641293832, 282.7433172575688])


def test_run_detection_bad_path_error(tmp_path):
    """
    test that ``run_detection`` raises FileNotFoundError
    if provided an unreadable image filepath
    """
    det_dir = tmp_path / "det"
    det_dir.mkdir(parents=True)
    ds = {"id": "BS3", "method": "BubbleSAM", "detection": {"img_dir": "bad/path"}}
    paths = {"det_dir": det_dir}
    with pytest.raises(FileNotFoundError, match="Invalid filepath."):
        wf.run_detection(ds, paths)
        
@pytest.mark.parametrize("method",
    ["bubblesam", "opencv"]
)
def test_stage_detect_returns_empty_dataframe(
    tmp_path: Path,
    method
):
    """
    test that stage detect returns an empty dataframe
    when the output of ``run_detection`` is `None` as
    a result of one of several dataset related warnings
    being raised. Empty Dataframe is propagated through
    ``stage_detect`` to ``main``, where a warning about
    the empty return is raised
    """
    ds = {"id": "DS8", "method": method}
    paths = {"proc_dir": tmp_path / "p", "det_dir": tmp_path / "d"}
    
    df_out = wf.stage_detect(ds, paths)
    assert df_out.empty

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
