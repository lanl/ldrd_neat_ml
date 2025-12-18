import pytest
import numpy as np
import pandas as pd
from importlib import resources
from typing import Generator, Any
import pooch  # type: ignore[import-untyped]
from matplotlib import rcParams
import requests
import os

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
    base_url = "doi:10.5281/zenodo.17545141",
    path = pooch.os_cache("test_images")
) 

@pytest.fixture(scope="session")
def reference_images():
    try:
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
    except ValueError:
        output_paths = []
        file_save_path = pooch.os_cache("test_images")
        # try manual download from figshare backup
        image_files_backup = requests.get("https://api.figshare.com/v2/articles/30546491")
        image_files_info = image_files_backup.json()
        image_files_data = image_files_info.get("files")
        for image_file in image_files_data:
            file_name = image_file["name"]
            download_url = image_file["download_url"]
            image_download = requests.get(download_url)
            output_path = os.path.join(file_save_path, file_name)
            output_paths.append(output_path)
            if not os.path.exists(output_path):
                with open(output_path, "wb") as f:
                    for out_file in image_download.iter_content():
                        f.write(out_file)  
        # reorder paths based on pooch order
        output_imgs = tuple(output_paths[i] for i in [0, 3, 2, 1])        
        return output_imgs

    return (images_Processed_raw,
            detection_raw,
            detection_processed,
            images_raw)
