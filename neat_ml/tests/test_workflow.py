import logging
from pathlib import Path
import pytest
import os
import shutil
import torch
import pandas as pd
from numpy.testing import assert_allclose
import re
from typing import Any, Dict

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

@pytest.mark.parametrize(
    ("steps_str", "expected"),
    [
        ("all", ["detect", "analysis", "train", "infer", "explain", "plot"]),
        (" Detect ,  Analysis ", ["detect", "analysis"]),  # whitespace + case normalization
        ("ANALYSIS,DETECT", ["analysis", "detect"]),  # preserves order after lowercasing
        ("", []),  # empty input -> empty list
        (", ,", []),  # only commas/whitespace -> empty list
        ("ALL", ["all"]),  # case-sensitive: 'ALL' does not expand
        ("detect,", ["detect"]),  # trailing comma ignored
        ("X,DETECT", ["x", "detect"]),  # unknown steps pass through lowercased
    ],
)
def test_as_steps_set_normalizes_and_expands(steps_str: str, expected: list[str]) -> None:
    """
    ``as_steps_set``: normalizes case/whitespace, preserves order, expands exact 'all',
    and passes unknown tokens through in lowercase.
    """
    assert wf.as_steps_set(steps_str) == expected

def test_get_path_structure_builds_expected_paths(tmp_path: Path):
    """
    get_path_structure: builds proc_dir and det_dir using ds_id/method/class/time_label.
    """
    roots = {"work": tmp_path, "results": tmp_path / "results"}
    ds = {
        "id": "DS1",
        "method": "OpenCV",
        "class": "pos",
        "time_label": "T01",
        "analysis": {
            "composition_csv": "comp.csv"
        }
    }
    steps = ['detect','analysis']

    paths = wf.get_path_structure(roots, ds, steps)  #type: ignore[arg-type]

    base = tmp_path / "DS1" / "OpenCV" / "pos" / "T01"
    assert paths["proc_dir"] == base / "T01_Processed_OpenCV"
    assert paths["det_dir"] == base / "T01_Processed_OpenCV_With_Blob_Data"

    # Default analysis outputs
    assert paths["per_csv"] == tmp_path / "results" / "DS1" / "per_image.csv"
    assert paths["agg_csv"] == tmp_path / "results" / "DS1" / "aggregate.csv"
    assert paths["composition_csv"] == Path("comp.csv")

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
                    not (torch.backends.mps.is_available()
                    or torch.cuda.is_available()), reason="only run when gpu available"
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
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path
):
    """
    stage_analyze_features: logs warning when neither
    analysis.input_dir nor paths['det_dir'] is available.
    """
    caplog.set_level(logging.WARNING)
    ds = {"id": "AN1", "method": "OpenCV", "time_label": "T01", "analysis": {}}
    wf.stage_analyze_features(ds, {})
    assert "No analysis input_dir provided and det_dir unavailable." in caplog.text


def test_stage_analyze_features_errors_when_input_dir_missing(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path
):
    """
    stage_analyze_features: logs warning if input_dir path does not exist.
    """
    caplog.set_level(logging.WARNING)
    input_dir = tmp_path / "no_such_dir"
    ds = {
        "id": "AN2",
        "method": "OpenCV",
        "time_label": "T01",
        "analysis": { 
            "input_dir": input_dir,
            "graph_method": "knn",
            "graph_param": 1
        }
    }

    wf.stage_analyze_features(ds, {})

    assert f"Analysis input_dir '{input_dir}' does not exist for 'AN2'." in caplog.text


def test_stage_analyze_features_errors_when_composition_csv_missing(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path
):
    """
    stage_analyze_features: logs warning if composition_csv is provided but does not exist.
    """
    caplog.set_level(logging.WARNING)
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    missing_csv = tmp_path / "missing.csv"

    ds = {
        "id": "AN3",
        "method": "OpenCV",
        "time_label": "T01",
        "analysis": {
            "input_dir": input_dir, 
            "composition_csv": missing_csv,
            "graph_method": "knn",
            "graph_param": 1
            }
         }
    wf.stage_analyze_features(ds, {})

    assert f"Composition CSV '{missing_csv}' missing for 'AN3'." in caplog.text


def test_stage_analyze_features_happy_path_calls_full_analysis(
    tmp_path: Path,
    mock_dir,
):
    """
    stage_analyze_features: happy path creates output dirs
    and calls full_analysis with expected args.
    """
    input_dir, output_dir, comp_csv = mock_dir
    out_per = output_dir / "per_image.csv"
    out_agg = output_dir / "aggregate.csv"
    
    ds = {
        "id": "AN4",
        "method": "OpenCV",
        "time_label": "T99",
        "composition_cols": ["PEG", "Dex"],
        "graph_method": "knn",
        "graph_param": 7,
        "analysis": {
            "input_dir": input_dir,
            "per_image_csv": out_per,
            "aggregate_csv": out_agg,
        },
    }
    roots = {"work": input_dir, "results": output_dir}
    paths = wf.get_path_structure(roots, ds, ["analysis"])
    wf.stage_analyze_features(ds, paths)

    # assertions about the outputs from calling ``full_analysis``
    df_per = pd.read_csv(out_per) 
    df_agg = pd.read_csv(out_agg)
    assert df_per.shape == (1, 30)
    assert df_agg.shape == (1, 95)


def test_stage_analyze_features_raises_when_input_dir_unavailable(
    caplog: pytest.LogCaptureFixture,
):
    """
    No analysis.input_dir and no paths['det_dir'] -> logs (now raises) and returns.
    """

    caplog.set_level(logging.WARNING)
    ds = {"id": "AN1", "method": "OpenCV", "time_label": "T01", "analysis": {}}
    wf.stage_analyze_features(ds, paths={})
    assert "No analysis input_dir provided and det_dir unavailable" in caplog.text


def test_stage_analyze_features_raises_when_input_dir_path_missing(
    tmp_path,
    caplog: pytest.LogCaptureFixture,
):
    """
    analysis.input_dir exists as a string but the path itself does not exist.
    """

    caplog.set_level(logging.WARNING)
    missing_dir = tmp_path / "no_such_dir"
    ds = {
        "id": "AN2",
        "method": "OpenCV",
        "time_label": "T01",
        "analysis": {
            "input_dir": missing_dir,
            "graph_method": "knn",
            "graph_param": 1
        },
    }
    wf.stage_analyze_features(ds, paths={})
    assert "Analysis input_dir" in caplog.text

def test_stage_analyze_features_raises_when_composition_csv_missing(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path
):
    """
    composition_csv provided but the file does not exist.
    """
    caplog.set_level(logging.WARNING)

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    missing_csv = tmp_path / "missing.csv"

    ds = {
        "id": "AN3",
        "method": "OpenCV",
        "time_label": "T01",
        "analysis": {
            "input_dir": input_dir,
            "composition_csv": missing_csv,
            "graph_method": "knn",
            "graph_param": 1
        },
    }
    wf.stage_analyze_features(ds, paths={})
    assert "Composition CSV" in caplog.text


@pytest.mark.parametrize("mode", ["OpenCV", "BubbleSAM"])
def test_stage_analyze_features_raises_when_no_detection_outputs_opencv(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
    mode,
):
    """
    Input dir exists but contains no parquet files.
    """
    caplog.set_level(logging.WARNING)
    input_dir = tmp_path / "empty_in"
    input_dir.mkdir()

    ds = {
        "id": "AN4",
        "method": mode,
        "time_label": "T01",
        "analysis": {
            "input_dir": input_dir,
            "graph_method": "knn",
            "graph_param": 1
        },
    }
    wf.stage_analyze_features(ds, paths={})
    assert "No detection outputs matching" in caplog.text


def test_stage_analyze_features_logs_when_input_dir_falsy_string(
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.WARNING)
    ds = {"id": "AN1", "method": "OpenCV", "time_label": "T01", "analysis": {}}
    wf.stage_analyze_features(ds, paths={})
    assert "No analysis input_dir provided" in caplog.text


@pytest.mark.parametrize("graph_method, graph_param, err_msg",
    [
        (None, None, "Please provide `graph_method` input."),
        ("knn", None, "Graph method:")
    ]
)
def test_stage_analyze_features_no_graph_method_param_error(
    tmp_path,
    graph_method,
    graph_param,
    err_msg
):
    """
    assert that a ValueError is raised when no ``graph_method`` is provided
    OR when ``graph_method`` is "knn" or "radius" and no ``graph_param`` is
    provided.
    """
    ds = {
        "id": "AN5",
        "method": "BubbleSAM",
        "time_label": "T01",
        "analysis":
            {
                "input_dir": tmp_path,
                "graph_method": graph_method,
                "graph_param": graph_param,
            },
        }
    with pytest.raises(ValueError, match=err_msg):
        wf.stage_analyze_features(ds, paths={})


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
