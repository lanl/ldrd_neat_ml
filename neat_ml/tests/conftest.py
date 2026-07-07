import pytest
import numpy as np
import pandas as pd
from importlib import resources
from typing import Generator, Any
import pooch  # type: ignore[import-untyped]
from matplotlib import rcParams
from pathlib import Path
import cv2

# try setting plot font to ``Arial``, if installed, 
# otherwise default to standard matplotlib font
rcParams['font.sans-serif'] = ["Arial"]
rcParams['font.family'] = "sans-serif"


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """Deterministic (seeded) composition/phase dataframe."""
    rng = np.random.default_rng(7)
    x = rng.uniform(0.0, 20.0, 30)
    y = rng.uniform(0.0, 20.0, 30)
    phase = (x + y > 20.0).astype(int)
    return pd.DataFrame({"Sodium Citrate (wt%)": x,
                         "PEO 8 kg/mol (wt%)": y,
                         "Phase": phase})

@pytest.fixture(scope="session")
def baseline_dir() -> Generator[Any, Any, Any]:
    """
    Directory that stores the reference (expected) images.
    """
    ref = resources.files("neat_ml.tests") / "baseline"
    with resources.as_file(ref) as path:
        yield path

# download testing image files with pooch
image_files = pooch.create(
    base_url = "https://github.com/lanl/ldrd_neat_ml_images/raw/main/test_images",
    path = pooch.os_cache("test_images"),
    registry = {
        "images_Processed_raw.tiff": "sha256:cc145dde89791119f59c3aaa75d12e1f23c2da6d87f02eabde50b2ab7ea648cb",
        "images_raw.tiff": "sha256:9ac34743ceb6449f93888e668cb6c1f79470434efd46aeedb6b9185173e6725b",
        "raw_detection.png": "sha256:496af9670a23c998cb50a6e00c46b46f915ea4110b4b62a382f901f5ad469d19",
        "raw_processed.png": "sha256:c5d9cdd527f87eb2d198d2d28717c40c5064ef0848707a7eb4f5f9e8a9f5b3c6",        
    }
) 

@pytest.fixture(scope="session")
def reference_images():
    images_Processed_raw = image_files.fetch(
        fname="images_Processed_raw.tiff",
    )
    detection_raw = image_files.fetch(
        fname="raw_detection.png",
    ) 
    detection_processed = image_files.fetch(
        fname="raw_processed.png",
    )
    images_raw = image_files.fetch(
        fname="images_raw.tiff", 
    )
    
    return (images_Processed_raw,
            detection_raw,
            detection_processed,
            images_raw)
         

@pytest.fixture(scope="session")
def mask_settings():
    return {
        "points_per_side": 4,
        "points_per_batch": 4,
        "pred_iou_thresh": 0.80,
        "stability_score_thresh": 0.80,
        "stability_score_offset": 0.1,
        "crop_n_layers": 1,
        "box_nms_thresh": 0.1,
        "crop_n_points_downscale_factor": 1,
        "min_mask_region_area": 5,
        "use_m2m": True,
    }


@pytest.fixture(scope="session")
def image_with_circles_fixture(tmp_path_factory) -> Path:
    """
    Return a path to a 100x100 black RGB image with two white circles.
    """
    img = np.zeros((100, 100, 3), np.uint8)
    white, filled = (255, 255, 255), -1
    cv2.circle(img, center=(30, 30), radius=10, color=white, thickness=filled)
    cv2.circle(img, center=(70, 65), radius=15, color=white, thickness=filled)
    fpath = tmp_path_factory.mktemp("imgs") / "circles.tiff"
    cv2.imwrite(fpath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return fpath


@pytest.fixture(scope="session")
def make_dummy_blobs():
    def _make(mode="opencv"):
        """
        Dataframe containing random values for the metrics of
        10 ``blobs`` to recapitulate the values found in the
        output parquet files from the blob detection step.
        """
        rng = np.random.default_rng(1)
        center_x = rng.integers(20, 100, 10)
        center_y = rng.integers(20, 100, 10)
        areas   = rng.integers(100, 500, 10)
        radii   = rng.integers(0, 10, 10)
        xmin = center_x - radii
        ymin = center_y - radii
        xmax = center_x + radii
        ymax = center_y + radii

        bboxes = np.column_stack((xmin, ymin, xmax, ymax)).tolist()
        if mode == "bubblesam":
            bboxes = [str(row) for row in bboxes]

        df = pd.DataFrame(
            {
                "center_x": center_x,
                "center_y": center_y,
                "area": areas,
                "radius": radii,
                "bbox": bboxes
            }
        )
        return df, center_x, center_y, areas, radii
    return _make

@pytest.fixture(scope="function")
def mock_dir(tmp_path_factory, make_dummy_blobs):
    """Creates a mock directory structure for end-to-end pipeline testing."""
    tmp_out_path = tmp_path_factory.mktemp("out")
    input_dir = tmp_out_path / "input"
    output_dir = tmp_out_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    ocv_fname = (
        "offset -5_bottom_A2_O_Ph_Raw_163c48ec-5ec9"
        "-4b1c-b304-ea40e77f0780_bubble_data.parquet.gzip"
    )
    df_ocv, _, _, _, _ = make_dummy_blobs("opencv")
    # recapituale ``center`` tuple handling
    df_ocv["center"] = list(zip(df_ocv["center_x"], df_ocv["center_y"]))
    df_ocv.to_parquet(input_dir / ocv_fname)

    bsam_fname = (
        "offset -5_bottom_A1_O_Ph_Raw_b96c0d64-03fd-"
        "4285-824d-e82eafedce90_masks_filtered.parquet.gzip"
    )
    df_bsam, _, _, _, _ = make_dummy_blobs("bubblesam")
    df_bsam.to_parquet(input_dir / bsam_fname)

    comp_df = pd.DataFrame({
        "UniqueID": ["163c48ec-5ec9-4b1c-b304-ea40e77f0780",
            "b96c0d64-03fd-4285-824d-e82eafedce90"],
        "Phase_Separation": [True, False],
        "Group": ["G1", "G2"],
    })
    comp_csv = tmp_out_path / "composition.csv"
    comp_df.to_csv(comp_csv, index=False)

    return input_dir, output_dir, comp_csv


@pytest.fixture(scope="session")
def real_blobs():
    """returns a dataframe of real blob data"""
    blob_dict = {
        'center_x': {
            0: 2220.5,
            1: 2120.5,
            2: 2377.0,
            3: 1998.5,
            4: 2423.5
        },
        'center_y': {
            0: 921.5,
            1: 739.5,
            2: 1098.5,
            3: 745.5,
            4: 719.5
        },
        'area': {
            0: 2082.0,
            1: 655.0,
            2: 845.0,
            3: 753.0,
            4: 808.0
        },
        'radius': {
            0: 25.743372,
            1: 14.439286,
            2: 16.400361,
            3: 15.481839,
            4: 16.037281
        },
        'bbox': {
            0: [896, 2194, 947, 2247],
            1: [725, 2106, 754, 2135],
            2: [1082, 2361, 1115, 2393],
            3: [730, 1983, 761, 2014],
            4: [704, 2407, 735, 2440]
        }
    }
    return pd.DataFrame(blob_dict)
