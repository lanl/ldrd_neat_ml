import pytest
import numpy as np
import pandas as pd
from importlib import resources
from typing import Generator, Any
import pooch  # type: ignore[import-untyped]


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
    base_url = "doi:10.5281/zenodo.17545141",
    path = pooch.os_cache("test_images")
)

@pytest.fixture(scope="session")
def reference_images():
    image_files.load_registry_from_doi()
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
