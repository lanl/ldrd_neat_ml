"""
Pre-processing utilities for TIFF images.

The module offers two public functions:

    - ``process_image``: CLAHE contrast enhancement followed by un-sharp masking.

    - ``process_directory``: end-to-end batch processor that writes enhanced
                             images to a target folder.

All heavy lifting happens in process_image; the other helpers are mere
orchestration wrappers.
"""
from pathlib import Path
import cv2
import numpy as np
import logging
from tqdm.auto import tqdm
from itertools import chain


log = logging.getLogger(__name__)

__all__ = [
    "process_image",
    "process_directory",
]

def process_image(
    img: np.ndarray,
    *,
    clip_limit: float,
    tile_grid_size: tuple[int, int],
    ksize: tuple[int, int],
    alpha: float,
) -> np.ndarray:
    """
    Enhance local contrast with CLAHE, then sharpen the image.

    Parameters
    ----------
    img : np.ndarray
        Single-channel 8-bit image loaded via cv2.IMREAD_GRAYSCALE.
    clip_limit : float
        Higher values increase contrast; values ≈ 2 are usually safe.
    tile_grid_size : tuple[int, int]
        Size (cols, rows) of the CLAHE tiles.
    ksize : tuple[int, int]
        Gaussian kernel for the un-sharp mask.
    alpha : float
        Sharpening strength; 0 disables sharpening.

    Returns
    -------
    sharpened : np.ndarray
        Sharpened image with identical shape and dtype as img.
    """
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit, 
        tileGridSize=tile_grid_size
    )
    enhanced = clahe.apply(img)
    blurred = cv2.GaussianBlur(
        enhanced, 
        ksize, 
        sigmaX=0
    )
    sharpened = cv2.addWeighted(
        enhanced, 
        1.0 + alpha, 
        blurred, 
        -alpha, 
        0)
    
    return sharpened


def process_directory(
    input_path: Path, 
    output_dir: Path
) -> None:
    """
    Run CLAHE + un-sharp masking on every TIFF found in input_path
    or any single TIFF image path provided.

    Note: The processed images are written to ``output_dir``
    so ``input_path`` sub-directory structure is not preserved
    and therefore identical filenames will be overwritten.

    Parameters
    ----------
    input_path : Path
        Path to directory that contains raw .tiff/.tif files
        or single .tiff/.tif image file path.
    output_dir : Path
        Destination folder; will be created if missing.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = []
    # check if a single input image is provided
    if input_path.is_file() and input_path.suffix.lower() in [".tiff", ".tif"]:
        input_paths.append(input_path)
    # otherwise find all the `.tiff` or `.tif` files recursively in the path
    else:
        input_paths.extend(list(chain(
            input_path.glob("**/*.tiff"),
            input_path.glob("**/*.tif"))
            )
        )
    # if no files are found in the provided path, raise ``FileNotFoundError``
    if not input_paths:
        raise FileNotFoundError(f"No `.tiff` or `.tif` files found in {input_path}")
    # iterate through images and perform preprocessing
    for in_path in tqdm(
        input_paths,
        total=len(input_paths),
        desc="Preprocessing Images",
    ):
        out_path = output_dir / in_path.name
        img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)  # type: ignore[call-overload]
        if img is not None:
            processed = process_image(
                img,
                clip_limit=2.0,
                tile_grid_size=(8,8),
                ksize=(5,5),
                alpha=1.5
            )
            cv2.imwrite(out_path, processed)  # type: ignore[call-overload]
        else:
            log.warning(f"Could not read file, skipping: {in_path}")
    log.info(f"Completed: (output: {output_dir.resolve()})")
