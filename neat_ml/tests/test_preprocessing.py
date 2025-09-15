from pathlib import Path
from importlib import resources
import cv2
import matplotlib
import re
import pytest
from pytest import MonkeyPatch
from typing import Any

from matplotlib.testing.compare import compare_images

from neat_ml.opencv import preprocessing as pp

matplotlib.use("Agg")

def test_process_directory_single_image(
    tmp_path: Path
):
    pkg_root = resources.files(__package__)
    data_res = pkg_root/"data"/"images"
    baseline_res = pkg_root/"baseline"

    with (
        resources.as_file(data_res) as test_data_dir,
        resources.as_file(baseline_res) as baseline_dir,
    ):        
        raw_input = test_data_dir / "raw.tiff"
        desired_png = baseline_dir / "raw_processed.png"
        
        pp.process_directory(test_data_dir, tmp_path)
        processed_tiff = tmp_path / raw_input.name
        img = cv2.imread(str(processed_tiff), cv2.IMREAD_GRAYSCALE)
        actual_png = tmp_path / "raw_processed.png"
        cv2.imwrite(str(actual_png), img)
        
        compare_images(str(desired_png), str(actual_png), tol=1e-4)

def test_process_directory_warns_on_unreadable_file(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input_images"
    output_dir = tmp_path / "processed_images"
    input_dir.mkdir()
    bad_img = input_dir / "bad.tif"
    bad_img.write_bytes(b"not-a-real-image")

    original_imread = cv2.imread

    def _fake_imread(path: Any, flags: int) -> Any:
        if str(Path(path)) == str(bad_img):
            return None
        return original_imread(path, flags)

    monkeypatch.setattr(cv2, "imread", _fake_imread, raising=True)
    pp.process_directory(input_dir, output_dir)
    warn_pat = rf"\[WARNING\] Could not read file, skipping: {re.escape(str(bad_img))}"
    pytest.raises(AssertionError, match=warn_pat)
