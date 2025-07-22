"""
Pre-processing utilities for TIFF images.

The module offers three public functions:

    process_image   - CLAHE contrast enhancement followed by un-sharp masking.
    iter_images     - lazy generator that yields every TIFF under a root path.
    process_directory   - end-to-end batch processor that writes enhanced
  images to a target folder.

All heavy lifting happens in process_image; the other helpers are mere
orchestration wrappers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Tuple
import cv2
import numpy as np

SUPPORTED_EXTS: Tuple[str, ...] = (".tiff", ".tif")

__all__ = [
    "SUPPORTED_EXTS",
    "process_image",
    "iter_images",
    "process_directory",
]

def process_image(
    img: np.ndarray,
    *,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    ksize: Tuple[int, int] = (5, 5),
    alpha: float = 1.5,
) -> np.ndarray:
    """
    Enhance local contrast with CLAHE, then sharpen the image.

    Parameters
    ----------
    img : np.ndarray
        Single-channel 8-bit image loaded via cv2.IMREAD_GRAYSCALE.
    clip_limit : float, default 2.0
        Higher values increase contrast; values ≈ 2 are usually safe.
    tile_grid_size : tuple[int, int], default (8, 8)
        Size (cols, rows) of the CLAHE tiles.
    ksize : tuple[int, int], default (5, 5)
        Gaussian kernel for the un-sharp mask.
    alpha : float, default 1.5
        Sharpening strength; 0 disables sharpening.

    Returns
    -------
    np.ndarray
        Image with identical shape and dtype as  img .
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


def iter_images(root: Path) -> Iterable[Path]:
    """
    Recursively yield all files under  root  whose extension is TIFF.

    Parameters
    ----------
    root : Path
        Directory that is walked depth-first.

    Returns
    -------
    Iterable[pathlib.Path]
        Absolute paths of discovered images.
    """
    for dirpath, _, files in os.walk(root):
        for name in files:
            if name.lower().endswith(SUPPORTED_EXTS):
                yield Path(dirpath) / name


def process_directory(
    input_dir: Path, 
    output_dir: Path
) -> None:
    """
    Run CLAHE + un-sharp masking on every TIFF found in  input_dir .

    The processed images are written to  output_dir
    sub-directory structure is not preserved, so identical
    filenames will be overwritten.

    Parameters
    ----------
    input_dir : Path
        Folder tree that contains raw .tiff / .tif files.
    output_dir : Path
        Destination folder; will be created if missing.

    Returns
    -------
    None
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for in_path in iter_images(input_dir):
        out_path = output_dir / in_path.name
        img = cv2.imread(str(in_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            processed = process_image(img)
            cv2.imwrite(str(out_path), processed)
        else:
            print(f"\n[WARNING] Could not read file, skipping: {in_path}")
    print(f"\n[INFO] Completed: (output: {output_dir.resolve()})")
