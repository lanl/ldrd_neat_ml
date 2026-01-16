from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import cv2
matplotlib.use("Agg")
import pytest
from matplotlib.testing.compare import compare_images
import pooch  # type: ignore[import-untyped]
from numpy.testing import assert_allclose

from neat_ml.opencv.detection import (
    collect_tiff_paths,
    _detect_single_image,
    _save_debug_overlay,
    run_opencv,
)

def test_visual_regression_debug_overlay(
    tmp_path: Path,
    reference_images: tuple,
):
    img_path = collect_tiff_paths(pooch.os_cache("test_images"))[1]
    stem = Path(img_path).stem

    actual_dir = tmp_path / "overlay"
    df_in = pd.DataFrame({"image_filepath": [img_path]})
    run_opencv(df=df_in, output_dir=actual_dir, debug=True)

    actual_png  = actual_dir/f"{stem}_debug.png"
    desired_png = reference_images[1]

    result = compare_images(
        desired_png,
        actual_png,
        tol=1e-4)  # type: ignore[call-overload]
    assert result is None


def test_detect_single_image_missing_file(tmp_path: Path) -> None:
    bogus = tmp_path / "does_not_exist.tiff"
    pattern = f"Unable to read image file: {bogus}"
    with pytest.raises(FileNotFoundError, match=pattern):
        _detect_single_image(str(bogus))

def test_run_opencv_missing_column(tmp_path: Path) -> None:
    df_bad = pd.DataFrame({"wrong": ["foo.tiff"]})
    pattern = "DataFrame must contain 'image_filepath' column."
    with pytest.raises(ValueError, match=pattern):
        run_opencv(df_bad, output_dir=tmp_path)

def test_save_debug_overlay_error(tmp_path: Path):
    bogus = tmp_path / "no_image.tiff"
    with pytest.raises(FileNotFoundError, match="Could not read image for debug overlay"):
        _save_debug_overlay(str(bogus), pd.DataFrame(), tmp_path)

def test_detect_single_image_no_blobs(tmp_path: Path):
    """
    Solid-black image should produce zero keypoints.
    """
    blank = np.zeros((100, 100), dtype=np.uint8)
    img_path = tmp_path / "blank.tiff"
    cv2.imwrite(img_path, blank)  # type: ignore[call-overload]

    num_blobs, median_r, bubble_data = _detect_single_image(str(img_path))
    assert num_blobs == 0
    assert np.isnan(median_r)
    assert bubble_data.empty
   
def test_detect_single_image_processed(tmp_path: Path, reference_images: tuple):
    """
    regression test for detection of keypoints in processed image
    """ 
    img_path = collect_tiff_paths(pooch.os_cache("test_images"))[1]
    num_blobs, median_r, bubble_data = _detect_single_image(str(img_path))
    assert num_blobs == 1735
    assert_allclose(median_r, 3.623063564300537)
    assert bubble_data.shape == (1735, 5)
