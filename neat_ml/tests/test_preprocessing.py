from pathlib import Path
import cv2
import matplotlib
import pytest
from pytest import MonkeyPatch
from typing import Any
import pooch  # type: ignore[import-untyped]
import os
import logging

from matplotlib.testing.compare import compare_images

from neat_ml.opencv import preprocessing as pp

matplotlib.use("Agg")

def test_process_directory_single_image(
    tmp_path: Path,
    reference_images : list,
):
    raw_input = reference_images[3] 
    pp.process_directory(pooch.os_cache("test_images"), tmp_path)
    processed_tiff = tmp_path / os.path.basename(raw_input)
    img = cv2.imread(str(processed_tiff), cv2.IMREAD_GRAYSCALE)
    actual_png = tmp_path / "raw_processed.png"
    assert img is not None
    cv2.imwrite(str(actual_png), img)
    
    result = compare_images(str(reference_images[2]), str(actual_png), tol=1e-4)
    assert result is None

def test_process_directory_warns_on_unreadable_file(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)
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
    assert "Could not read file, skipping:" in caplog.text
