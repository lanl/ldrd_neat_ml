import pytest
import numpy as np
import pandas as pd
from importlib import resources
from typing import Generator, Any
import pooch  # type: ignore[import-untyped]
from matplotlib import rcParams
from pathlib import Path
import cv2
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
from sklearn.datasets import make_classification

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

    df = pd.DataFrame(
        {
            "center_x": center_x,
            "center_y": center_y,
            "area": areas,
            "radius": radii,
            "bbox_xmin": xmin,
            "bbox_xmax": xmax,
            "bbox_ymin": ymin,
            "bbox_ymax": ymax,
        }
    )
    return df, center_x, center_y, areas, radii

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
    df, _, _, _, _ = make_dummy_blobs
    df.to_parquet(input_dir / ocv_fname)

    bsam_fname = (
        "offset -5_bottom_A1_O_Ph_Raw_b96c0d64-03fd-"
        "4285-824d-e82eafedce90_masks_filtered.parquet.gzip"
    )
    df.to_parquet(input_dir / bsam_fname)

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
        "bbox_xmin": {
            0: 896,
            1: 725,
            2: 1082,
            3: 730,
            4: 704
        },
        "bbox_xmax": {
            0: 2194,
            1: 2106,
            2: 2361,
            3: 1983,
            4: 2407
        },
        "bbox_ymin": {
            0: 947,
            1: 754,
            2: 1115,
            3: 761,
            4: 735
        },
        "bbox_ymax": {
            0: 2247,
            1: 2135,
            2: 2393,
            3: 2014,
            4: 2440
        },
    }
    return pd.DataFrame(blob_dict)

@pytest.fixture(scope="session")
def stable_rc():
    STABLE_RC = {
        "figure.figsize": (6.0, 4.0),
        "figure.dpi": 100,
        "savefig.dpi": 100,
        "savefig.bbox": "standard",
        "savefig.pad_inches": 0.0,
        "font.family": ["DejaVu Sans"],
        "font.size": 10.0,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.linewidth": 1.0,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "path.simplify": False,
        "text.antialiased": True,
        "lines.antialiased": True,
    }
    return STABLE_RC


@pytest.fixture(scope="session")
def sample_data() -> pd.DataFrame:
    """
    Provides a sample DataFrame for consistent testing.
    """
    rng = np.random.default_rng(42)
    data = {
        "feature1": rng.random(100),
        "feature2": rng.random(100) * 10,
        "feature3": ["A"] * 50 + ["B"] * 50,
        "exclude_col": np.arange(100),
        "target": rng.integers(0, 2, 100),
    }
    df = pd.DataFrame(data)
    df.loc[5, "feature1"] = np.nan
    return df


@pytest.fixture(scope="session")
def trained_model_bundle(tmp_path_factory):
    """Creates and saves a dummy trained model bundle."""
    tmp_model_path = tmp_path_factory.mktemp("model")
    features = ["feat_a", "feat_b"]
    model = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(random_state=42)),
        ]
    )
    rng = np.random.default_rng(7)
    dummy_X = pd.DataFrame(rng.random((10, len(features))), columns=features)
    dummy_y = pd.Series(rng.integers(0, 2, 10))
    model.fit(dummy_X, dummy_y)

    bundle = {"model": model, "features": features}
    model_path = tmp_model_path / "model.joblib"
    joblib.dump(bundle, model_path)
    return model_path


@pytest.fixture(scope="session")
def sample_inference_data(tmp_path_factory):
    """
    Provides a sample CSV file for inference testing.
    """
    tmp_infer_path = tmp_path_factory.mktemp("infer")
    rng = np.random.default_rng(123)
    data = {
        "feat_a": rng.random(50),
        "feat_b": np.arange(50),
        "id_col": [f"id_{i}" for i in range(50)],
        "ground_truth": rng.integers(0, 2, 50),
    }
    df = pd.DataFrame(data)
    csv_path = tmp_infer_path / "inference_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(scope="module")
def classification_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic binary-classification data."""
    X_arr, y = make_classification(
        n_samples=10,
        n_features=5,
        n_informative=3,
        random_state=0
    )
    X = pd.DataFrame(
        X_arr,
        columns=[
            "PEO 10 kg/mol (wt%)",
            "Dextran 10 kg/mol (wt%)",
            "num_blobs",
            "coverage_percentage",
            "graph_num_components",
        ],
    )
    return X, pd.Series(y, name="y")
