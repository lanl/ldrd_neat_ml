from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pytest
from matplotlib.testing.compare import compare_images

from neat_ml.opencv.detection import (
    collect_tiff_paths,
    build_df_from_img_paths,
    run_opencv,
)

@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Directory that holds raw TIFFs to be analysed."""
    return Path(__file__).parent / "data" / "images_Processed"


@pytest.fixture(scope="session")
def baseline_dir() -> Path:
    """Ground-truth results produced by a trusted OpenCV run."""
    return Path(__file__).parent / "baseline"

def test_visual_regression_debug_overlay(
    tmp_path: Path, 
    data_dir: Path, 
    baseline_dir: Path
):
    img_path = collect_tiff_paths(data_dir)[0]
    stem     = Path(img_path).stem

    actual_dir = tmp_path / "overlay"
    df_in = build_df_from_img_paths([img_path])
    run_opencv(df=df_in, output_dir=actual_dir, debug=True)

    actual_png  = actual_dir / f"{stem}_debug.png"
    desired_png = baseline_dir  / f"{stem}_detection.png"

    assert actual_png.is_file(),  f"Result PNG missing:   {actual_png}"
    assert desired_png.is_file(), f"Baseline PNG missing: {desired_png}"

    compare_images(str(desired_png), str(actual_png), tol=10.0)