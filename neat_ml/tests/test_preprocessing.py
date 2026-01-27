from pathlib import Path
import cv2
import matplotlib
import pytest
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
    pp.process_directory(Path(raw_input), tmp_path)
    processed_tiff = tmp_path / os.path.basename(raw_input)
    img = cv2.imread(processed_tiff, cv2.IMREAD_GRAYSCALE)
    actual_png = tmp_path / "raw_processed.png"
    cv2.imwrite(actual_png, img)  # type: ignore[call-overload]
    
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


@pytest.mark.parametrize("input_path",
    [
        2,  # case where the file path suffix for a single file is not `.tiff` or `.tif`
        "no_files",  # case where there are no files found in a provided directory
    ]
)
def test_process_directory_error(tmp_path, reference_images, input_path):
    """
    check that FileNotFoundError raised when no .tiff/.tif files found
    """
    if type(input_path) is int:
        input_path = Path(reference_images[input_path])
    else:
        input_path = tmp_path / input_path
    with pytest.raises(FileNotFoundError, match="No `.tiff` or `.tif` files found"):
        pp.process_directory(input_path, tmp_path)
