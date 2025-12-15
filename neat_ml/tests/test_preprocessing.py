from pathlib import Path
import cv2
import matplotlib
import pytest
import pooch  # type: ignore[import-untyped]
import os
import logging

from matplotlib.testing.compare import compare_images

from neat_ml.opencv import preprocessing as pp

matplotlib.use("Agg")

def test_process_directory_single_image(
    tmp_path: Path,
    reference_images : tuple,
):
    raw_input = reference_images[3] 
    pp.process_directory(pooch.os_cache("test_images"), tmp_path)
    processed_tiff = tmp_path / os.path.basename(raw_input)
    img = cv2.imread(str(processed_tiff), cv2.IMREAD_GRAYSCALE)
    actual_png = tmp_path / "raw_processed.png"
    cv2.imwrite(str(actual_png), img)  # type: ignore[arg-type]
    
    result = compare_images(
        reference_images[2],
        actual_png,
        tol=1e-4)  # type: ignore[call-overload] 
    assert result is None

def test_process_directory_warns_on_unreadable_file(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.WARNING)
    input_dir = tmp_path / "input_images"
    output_dir = tmp_path / "processed_images"
    input_dir.mkdir()
    bad_img = input_dir / "bad.tif"
    bad_img.write_bytes(b"not-a-real-image")

    pp.process_directory(input_dir, output_dir)
    assert "Could not read file, skipping:" in caplog.text


@pytest.mark.parametrize("img_path",
    [
        "images_raw.tiff",
        "",
    ]
)
def test_iter_images_yields_path(img_path):
    """
    test functionality of ``iter_images`` to return path from
    either directory or single image file path.
    """
    base_path = pooch.os_cache("test_images")
    in_path = os.path.join(base_path, img_path)
    out_paths = pp.iter_images(in_path)
    for out_path in out_paths:
        assert os.path.isfile(out_path)
