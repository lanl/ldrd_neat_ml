import logging
from pathlib import Path
import pytest
import os
import shutil

import neat_ml.workflow.lib_workflow as wf


def test_get_path_structure_builds_expected_paths(tmp_path: Path):
    """
    get_path_structure: builds proc_dir and det_dir using ds_id/method/class/time_label.
    """
    roots = {"work": str(tmp_path)}
    ds = {"id": "DS1", "method": "OpenCV", "class": "pos", "time_label": "T01"}

    paths = wf.get_path_structure(roots, ds)

    base = tmp_path / "DS1" / "OpenCV" / "pos" / "T01"
    assert paths["proc_dir"] == base / "T01_Processed_OpenCV"
    assert paths["det_dir"] == base / "T01_Processed_OpenCV_With_Blob_Data"


def test_get_path_structure_missing_work_raises_keyerror(tmp_path: Path):
    """
    If 'work' key is missing in roots, a KeyError is raised.
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
    ds: dict,
    paths: dict,
    suff: str,
    method: str,
) -> None:
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
    mocker,
    mock_tiny_weights,
    reduced_mask_settings,
    reference_images: tuple,
    ds: dict,
    paths: dict,
    exp_columns: set,
) -> None:
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
            # move sam2 weights to working dir
            chkpt_dir = Path("neat_ml/sam2/checkpoints")
            chkpt_dir.mkdir(parents=True)
            shutil.copy(
                mock_tiny_weights,
                chkpt_dir,
            )  # type: ignore[call-overload]

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
                    'tif_img_debug.png',
                    'tif_img_bubble_data.parquet.gzip',
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
                    'tif_img_masks_filtered.parquet.gzip',
                    'tiff_img_masks_filtered.parquet.gzip',
                    'bubblesam_summary.csv',
                ]
            )
        )
    
    # here and above, check that the absolute file paths for ``proc_dir``
    # and ``det_dir`` are generated by ``stage_opencv``
    assert proc_dir_exp.is_absolute() 
    assert det_dir_exp.is_absolute()
    assert df_out is not None
    assert df_out.shape == (2, 3)
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


def test_run_workflow_single_image_path(
    tmp_path,
    image_with_circles_fixture,
    reduced_mask_settings,
    mock_tiny_weights
):
    """
    test that providing run workflow with a path to a single image
    generates the correct outputs
    """
    det_dir = tmp_path / "det"

    ds = {
        "id": "BS5",
        "method": "bubblesam",
        "detection": {"img_dir": image_with_circles_fixture}
    }
    paths = {"det_dir": det_dir}

    df_out = wf.run_detection(ds, paths)
    assert df_out.shape == (1, 3)
    assert (det_dir / "bubblesam_summary.csv").is_file()
    assert (det_dir / "circles_masks_filtered.parquet.gzip").is_file()


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
