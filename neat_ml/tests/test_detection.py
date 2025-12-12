from pathlib import Path
from importlib import resources
import re

import matplotlib
import numpy as np
import pandas as pd
import cv2
matplotlib.use("Agg")
import pytest
from matplotlib.testing.compare import compare_images

from neat_ml.opencv.detection import (
    collect_tiff_paths,
    _detect_single_image,
    build_df_from_img_paths,
    _save_debug_overlay,
    run_opencv,
)

def test_visual_regression_debug_overlay(
    tmp_path: Path
):
    pkg_root = resources.files(__package__)
    data_res = pkg_root/"data"/"images_Processed"
    baseline_res = pkg_root/"baseline"

    with (
        resources.as_file(data_res) as data_dir,
        resources.as_file(baseline_res) as baseline_dir,
    ):
        img_path = collect_tiff_paths(data_dir)[0]
        stem = Path(img_path).stem

        actual_dir = tmp_path / "overlay"
        df_in = build_df_from_img_paths([img_path])
        run_opencv(df=df_in, output_dir=actual_dir, debug=True)

        actual_png  = actual_dir/f"{stem}_debug.png"
        desired_png = baseline_dir/f"{stem}_detection.png"

        compare_images(str(desired_png), str(actual_png), tol=1e-4)

def test_build_df_no_paths_raises() -> None:
    pattern = re.escape("No image paths provided to build DataFrame.")
    with pytest.raises(ValueError, match=pattern):
        build_df_from_img_paths([])

def test_detect_single_image_missing_file(tmp_path: Path) -> None:
    bogus = tmp_path / "does_not_exist.tiff"
    pattern = rf"Unable to read image file: {re.escape(str(bogus))}"
    with pytest.raises(FileNotFoundError, match=pattern):
        _detect_single_image(str(bogus))

def test_run_opencv_missing_column(tmp_path: Path) -> None:
    df_bad = pd.DataFrame({"wrong": ["foo.tiff"]})
    pattern = re.escape("DataFrame must contain 'image_filepath' column.")
    with pytest.raises(ValueError, match=pattern):
        run_opencv(df_bad, output_dir=tmp_path)

def test_save_debug_overlay_warns(tmp_path: Path):
    bogus = tmp_path / "no_image.tiff"
    with pytest.warns(UserWarning, match="Could not read image for debug overlay"):
        _save_debug_overlay(str(bogus), [], tmp_path)

def test_detect_single_image_no_blobs(tmp_path: Path):
    """
    Solid-black image should produce zero keypoints.
    """
    blank = np.zeros((100, 100), dtype=np.uint8)
    img_path = tmp_path / "blank.tiff"
    assert cv2.imwrite(str(img_path), blank)

    num_blobs, median_r, bubble_data = _detect_single_image(str(img_path))
    assert num_blobs == 0
    assert np.isnan(median_r)
    assert len(bubble_data) == 1
    first = bubble_data[0]
    assert np.isnan(first["bubble_number"])
    assert np.isnan(first["radius"])
    assert all(np.isnan(c) for c in first["bbox"])
 
