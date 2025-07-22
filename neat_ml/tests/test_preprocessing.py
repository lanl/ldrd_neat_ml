from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import pytest
from matplotlib.testing.compare import compare_images

from neat_ml.opencv import preprocessing as pp

matplotlib.use("Agg")

@pytest.fixture(scope="session")
def baseline_dir() -> Path:
    return Path(__file__).parent / "baseline"

@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    return Path(__file__).parent / "data/images"

def test_process_directory_single_image(
    tmp_path: Path,
    test_data_dir: Path,
    baseline_dir: Path,
):
    raw_input      = test_data_dir / "raw.tiff"
    desired_png   = baseline_dir  / "raw_processed.png"
    assert raw_input.is_file(),   f"Missing test input: {raw_input}"
    assert desired_png.is_file(), f"Missing baseline:   {desired_png}"
    
    pp.process_directory(test_data_dir, tmp_path)
    processed_tiff = tmp_path / raw_input.name
    img = cv2.imread(str(processed_tiff), cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"cv2.imread failed for {processed_tiff}"
    
    actual_png = tmp_path / "raw_processed.png"
    cv2.imwrite(str(actual_png), img)
    
    compare_images(str(desired_png), str(actual_png), tol=10)