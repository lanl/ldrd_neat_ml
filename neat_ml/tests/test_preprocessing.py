from pathlib import Path
from importlib import resources

import cv2
import matplotlib
from matplotlib.testing.compare import compare_images

from neat_ml.opencv import preprocessing as pp

matplotlib.use("Agg")

def test_process_directory_single_image(
    tmp_path: Path
):
    with resources.as_file(resources.files(__package__) / "data" / "images") as test_data_dir, \
         resources.as_file(resources.files(__package__) / "baseline") as baseline_dir:
        
        raw_input = test_data_dir / "raw.tiff"
        desired_png = baseline_dir / "raw_processed.png"
        
        pp.process_directory(test_data_dir, tmp_path)
        processed_tiff = tmp_path / raw_input.name
        img = cv2.imread(str(processed_tiff), cv2.IMREAD_GRAYSCALE)
        actual_png = tmp_path / "raw_processed.png"
        cv2.imwrite(str(actual_png), img)
        
        compare_images(str(desired_png), str(actual_png), tol=1e-4)