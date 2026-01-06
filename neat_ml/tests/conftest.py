import pytest
import numpy as np
import pandas as pd
from importlib import resources
from typing import Generator, Any
import pooch  # type: ignore[import-untyped]
from matplotlib import rcParams

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
# image files stored at the following url:
# https://zenodo.org/records/17545141
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
         fname = "raw_processed.png",
    )
    images_raw = image_files.fetch(
         fname="images_raw.tiff", 
    )
    
    return (images_Processed_raw,
            detection_raw,
            detection_processed,
            images_raw)
