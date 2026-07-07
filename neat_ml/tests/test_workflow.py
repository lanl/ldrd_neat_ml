import logging
from pathlib import Path
import pytest
import os
import shutil
import torch
import pandas as pd
from numpy.testing import assert_allclose
import copy

import neat_ml.workflow.lib_workflow as wf

@pytest.mark.parametrize(
    ("steps_str", "expected", "err"),
    [
        ("all", ["detect", "analysis"], False),  # expands to full pipeline
        (" detect ,  analysis ", ["detect", "analysis"], False),  # whitespace removed
        ("ANALYSIS,DETECT", None, True),  # case sensitive inputs raise error
        ("", None, True),  # empty input -> empty list -> raises error
        ("ALL", None, True),  # 'ALL' raises error
        ("detect,", ["detect"], False),  # trailing comma ignored
        ("X,DETECT", None, True),  # unknown steps raise error
    ],
)
def test_as_steps_set_normalizes_and_expands(steps_str, expected, err):
    """
    test that ``as_steps_set`` returns appropriate steps without whitespace,
    expands exact 'all', and enforces lowercase user inputs. 
    """ 
    if err:
        with pytest.raises(ValueError, match="not contained in allowed steps"):
            wf.as_steps_set(steps_str)
    else:
        assert wf.as_steps_set(steps_str) == expected


@pytest.mark.parametrize("roots, ds, steps",
    [
        (
            {"work": ""},
            {},
            ["detect"],
        ),
        (
            {"work": "", "results": "results"},
            {
                "analysis": {
                    "composition_csv": "comp.csv"
                }
            },
            ["detect", "analysis"]
        ),
        (
            {"work": ""},
            {
                "analysis": {
                    "composition_csv": "comp.csv",
                    "per_image_csv" : "per_img.csv",
                    "aggregate_csv": "aggregate.csv",
                }
            },
            ["detect", "analysis"]
        ),
        (
            {"work": "", "results": "results"},
            {
                "analysis": {
                    "composition_csv": "comp.csv",
                    "per_image_csv" : "per_img.csv",
                }
            },
            ["detect", "analysis"]
        ),
        (
            {"work": "", "results": "results"},
            {
                "analysis": {
                    "per_image_csv" : "per_img.csv",
                }
            },
            ["detect", "analysis"]
        ),
    ],
)
def test_get_path_structure_builds_expected_paths(
    tmp_path: Path,
    roots: dict,
    ds: dict,
    steps: list,
):
    """
    test that `get_path_structure` builds the appropriate paths
    given the contents of the user input yaml file
    """
    base_ds = {
        "id": "DS1",
        "method": "OpenCV",
        "class": "pos",
        "time_label": "T01",
        "composition_csv": "comp.csv",
    }
    base_ds.update(ds)
    input_ds = copy.deepcopy(base_ds)
    roots = {k: tmp_path / v for k, v in roots.items()}
    paths = wf.get_path_structure(roots, input_ds, steps)

    base = tmp_path / "DS1" / "OpenCV" / "pos" / "T01"
    assert paths["proc_dir"] == base / "T01_Processed_OpenCV"
    assert paths["det_dir"] == base / "T01_Processed_OpenCV_With_Blob_Data"

    # Default analysis outputs
    if "analysis" in steps:
        analysis_dirs = base_ds.get("analysis")
        per_img_path = analysis_dirs.get("per_image_csv")  # type: ignore[union-attr]
        agg_path = analysis_dirs.get("aggregate_csv")  # type: ignore[union-attr]
        exp_per = (Path(per_img_path) if per_img_path is not None
            else tmp_path / "results" / "DS1" / "per_image.csv")
        exp_agg = (Path(agg_path) if agg_path is not None
            else tmp_path / "results" / "DS1" / "aggregate.csv")
        assert paths["per_csv"] == exp_per 
        assert paths["agg_csv"] == exp_agg
        assert paths["composition_csv"] == Path("comp.csv")

def test_get_path_structure_fallbacks(tmp_path: Path):
    """
    test that `get_path_structure` uses input dict fallbacks
    when not explicitly defined by user. will fail if fallbacks are None.
    """
    roots = {"work": "work_path"}
    ds = {"method": "OpenCV", "id": "ds_id"}
    paths = wf.get_path_structure(roots, ds, ["detect"])
    assert paths.get("proc_dir") == Path("work_path/ds_id/OpenCV/_Processed_OpenCV")
    assert paths.get("det_dir") == Path("work_path/ds_id/OpenCV/_Processed_OpenCV_With_Blob_Data")

def test_get_path_structure_missing_work_raises_keyerror(tmp_path: Path):
    """
    If 'work' key is missing in roots, a KeyError is raised.
    """
    roots = {"result": str(tmp_path)}
    ds = {"id": "DS1", "method": "OpenCV", "class": "pos", "time_label": "T01"}
    steps = ['detect','analysis']

    with pytest.raises(KeyError, match="work"):
        wf.get_path_structure(roots, ds, steps)  # type: ignore[arg-type]

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

def test_stage_analyze_features_warns_when_input_dir_unavailable(
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


def test_stage_analyze_features_warns_when_composition_csv_missing(
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
            }
         }
    roots = {"work": "work_path", "results": "results_path"}
    paths = wf.get_path_structure(roots, ds, ["analysis"])
    wf.stage_analyze_features(ds, paths)

    assert f"Composition CSV '{missing_csv}' missing for 'AN3'." in caplog.text


@pytest.mark.parametrize("include_save_paths, include_ds_id, include_analysis_cfg, ds_id",
    [
        # test case with user provided save paths and `ds_id`
        # desired behavior: use user provided save paths
        (True, True, True, "AN4"),
        # test case without user provided save paths, but with `ds_id`
        # desired behavior: use default save paths with `ds_id` subdir
        (False, True, True, "AN4"),
        # test case without user provided save paths or `ds_id`
        # desired behavior: use default save paths with default `ds_id` ("unknown")
        (False, False, True, "unknown"),
        # test case without user provided `analysis` cfg
        # desired behavior: use default paths `det_dir` for `input_dir`
        (False, True, False, "AN4"),
    ]
)
def test_stage_analyze_features_happy_path_calls_full_analysis(
    tmp_path: Path,
    mock_dir: tuple[Path, Path, Path],
    include_save_paths: bool,
    include_ds_id: bool,
    include_analysis_cfg: bool,
    ds_id: str,
):
    """
    stage_analyze_features: happy path creates output dirs
    and calls full_analysis with expected args.
    """
    input_dir, output_dir, comp_csv = mock_dir
    
    ds = {
        "method": "OpenCV",
        "time_label": "T99",
        "composition_cols": ["PEG", "Dex"],
        "graph_method": "knn",
        "k_param": 7,
    }
    if include_analysis_cfg:
        ds.update(
            {
                "analysis": {
                    "input_dir": input_dir,
                }
            }
        )
    # user provided `ds_id` (overrides default "unknown")
    if include_ds_id:
        ds.update({"id": ds_id})
    # user provided save paths override default save paths
    if include_save_paths:
        out_per = output_dir / "per_image.csv"
        out_agg = output_dir / "aggregate.csv"
        ds["analysis"].update( # type: ignore[attr-defined]
            {
                "per_image_csv": out_per,
                "aggregate_csv": out_agg,
            }
        )
    else:
        # expected default save paths
        out_per = output_dir / ds_id / "per_image.csv"
        out_agg = output_dir / ds_id / "aggregate.csv"
    roots = {"work": str(input_dir), "results": str(output_dir)}
    paths = wf.get_path_structure(roots, ds, ["analysis"])
    if not include_analysis_cfg:
        # in this case, if no user `analysis` cfg is provided
        # the default input directory is set to the `det_dir`
        # path. Recapitulate this behavior by copying the files
        # from `mock_dir` to the expected path.
        shutil.copytree(input_dir, paths["det_dir"])

    wf.stage_analyze_features(ds, paths)

    # assertions about the outputs from calling ``full_analysis``
    df_per = pd.read_csv(out_per) 
    df_agg = pd.read_csv(out_agg)
    assert df_per.shape == (1, 29)
    assert df_agg.shape == (1, 91)
    # spot checks on ``df_per`` outputs
    assert_allclose(df_per["std_blob_area"], 107.07261295235324)
    assert_allclose(df_per["graph_degree_std"], 0.942809)
    assert_allclose(df_per["mean_voronoi_area"], 5450.5100956330925)
    # spot checks on ``df_agg`` outputs
    assert_allclose(df_agg["median_blob_area_max"], 358.0)
    assert_allclose(df_agg["graph_avg_clustering_median"], 0.8968253968253969)
    assert_allclose(df_agg["coverage_percentage_max"], 38.04123711340206)
    assert_allclose(df_agg["mean_voronoi_area_std"], 0.0)


@pytest.mark.parametrize("mode, input_exist, warn_msg",
    [
        ("OpenCV", True, "No detection outputs matching"),
        ("BubbleSAM", True, "No detection outputs matching"),
        ("OpenCV", False, "Analysis input_dir"),
    ]
)
def test_stage_analyze_features_input_dir_warnings(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
    mode: str,
    input_exist: bool,
    warn_msg: str,
):
    """
    Input dir exists but contains no parquet files.
    """
    caplog.set_level(logging.WARNING)
    input_dir = tmp_path / "input_dir"
    if input_exist:
        input_dir.mkdir()

    ds = {
        "id": "AN4",
        "method": mode,
        "time_label": "T01",
        "analysis": {
            "input_dir": input_dir,
            "graph_method": "knn",
            "k_param": 1
        },
    }
    wf.stage_analyze_features(
        ds,
        paths={"per_csv": Path("per_img.csv"), "agg_csv": Path("agg.csv")}
    )
    assert warn_msg in caplog.text


@pytest.mark.parametrize("graph_method, graph_param, err_msg",
    [
        (None, None, "Please provide `graph_method` input."),
        ("knn", None, "Graph method:"),
        ("radius", None, "Graph method:"),
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
    OR when ``graph_method`` is "knn" or "radius" and the appropriate parameter is
    not provided.
    """
    ds = {
        "id": "AN5",
        "method": "BubbleSAM",
        "time_label": "T01",
        "analysis":
            {
                "input_dir": tmp_path,
                "graph_method": graph_method,
                "k_param": graph_param,
                "r_param": graph_param,
            },
        }
    with pytest.raises(ValueError, match=err_msg):
        wf.stage_analyze_features(ds, paths={"per_csv": "per_img.csv", "agg_csv": "agg.csv"})
