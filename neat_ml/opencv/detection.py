from pathlib import Path
from typing import Any, Dict, Tuple, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.color
from joblib import Memory
from tqdm.auto import tqdm

import warnings

__all__: Sequence[str] = [
    "collect_tiff_paths", 
    "_detect_single_image",
    "build_df_from_img_paths", 
    "_save_debug_overlay",
    "run_opencv"
]

def collect_tiff_paths(
    root: str | Path
) -> list[str]:
    """
    Recursively locate every TIFF image beneath a given directory.

    Parameters
    ----------
    root : str or Path
        Root directory to search for files.

    Returns
    -------
    list[str]
        List of absolute POSIX-style file paths for each .tiff file found.
    """
    root_path = Path(root).expanduser().resolve()
    return [str(Path(p).resolve()) for p in root_path.glob("**/*.tiff")]


def build_df_from_img_paths(
    img_paths: list[str]
) -> pd.DataFrame:
    """
    Convert a list of image file paths into a DataFrame.

    Parameters
    ----------
    img_paths : list[str]
        List of absolute file paths to images.

    Returns
    -------
    pd.DataFrame
        DataFrame with a single column 'image_filepath' containing the paths.
    """
    if not img_paths:
        raise ValueError("No image paths provided to build DataFrame.")
    return pd.DataFrame({"image_filepath": img_paths})

def _detect_single_image(
    img_path: str
) -> Tuple[int, float, list[Dict[str, Any]]]:
    """
    Detect bubbles in a single image using OpenCV's SimpleBlobDetector.

    Parameters
    ----------
    img_path : str
        Absolute file path to the image to process.

    Returns
    -------
    Tuple[int, float, List[Dict[str, Any]]]
        num_blobs : int
            Number of blobs detected in the image.
        median_radius : float
            Median radius of detected blobs (NaN if none).
        bubble_data : list[Dict[str, Any]]
            List of dictionaries, one per detected blob, each containing:
            - 'bubble_number' (int)
            - 'center' (tuple[float, float])
            - 'radius' (float)
            - 'area' (float)
            - 'bbox' (tuple[int, int, int, int])
    """
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to read image file: {img_path}")

    params = cv2.SimpleBlobDetector_Params() # type: ignore[attr-defined]
    params.minThreshold = 10
    params.maxThreshold = 200
    params.thresholdStep = 10
    params.filterByColor = False
    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 50000
    params.filterByCircularity = True
    params.minCircularity = 0.75
    params.filterByConvexity = True
    params.minConvexity = 0.80
    params.filterByInertia = True
    params.minInertiaRatio = 0.75

    detector = cv2.SimpleBlobDetector_create(params) # type: ignore[attr-defined]
    keypoints = detector.detect(image)

    bubble_data: list[Dict[str, Any]] = []
    radii: list[float] = []

    if keypoints:
        for idx, kp in enumerate(keypoints):
            cx, cy = kp.pt
            r = kp.size / 2.0
            bbox = (cx - r, cy - r, cx + r, cy + r)
            bubble_data.append(
                {
                    "bubble_number": idx + 1,
                    "center": (cx, cy),
                    "radius": r,
                    "area": np.pi * r**2,
                    "bbox": bbox,
                }
            )
            radii.append(r)
    else:
        nan_bbox = (np.nan, np.nan, np.nan, np.nan)
        bubble_data.append(
            {
                "bubble_number": np.nan,
                "center": np.nan,
                "radius": np.nan,
                "area": np.nan,
                "bbox": nan_bbox,
            }
        )
        radii.append(np.nan)

    num_blobs = len(keypoints)
    median_radius = float(np.nanmedian(radii))
    return num_blobs, median_radius, bubble_data

def _save_debug_overlay(
    img_path: str,
    bubble_data: list[Dict[str, Any]],
    out_dir: Path,
) -> None:
    """
    Create and save a side-by-side original/overlay PNG.

    Parameters
    ----------
    img_path : str
        File path to the original image.
    bubble_data : list[Dict[str, Any]]
        List of bubble metadata as returned by _detect_single_image.
    out_dir : Path
        Directory where the debug PNG will be saved.

    Returns
    -------
    None
    """
    image_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if image_gray is None:
        warnings.warn(f"Could not read image for debug overlay: {img_path}")
        return
    image_rgb = skimage.color.gray2rgb(image_gray)

    overlay = image_gray.copy()
    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)

    for bubble in bubble_data:
        bbox = bubble["bbox"]
        if (
            not isinstance(bbox, (tuple, list))
            or len(bbox) != 4
            or any(not np.isfinite(c) for c in bbox)
        ):
            continue
        x_min, y_min, x_max, y_max = map(int, bbox)
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    ax[0].imshow(image_rgb)
    ax[0].set_title("Original")
    ax[0].axis("off")
    ax[1].imshow(overlay)
    ax[1].set_title("Detected blobs")
    ax[1].axis("off")

    png_name = f"{Path(img_path).stem}_debug.png"
    fig.savefig(out_dir / png_name, dpi=300, bbox_inches="tight")
    plt.close(fig)

def run_opencv(
    df: pd.DataFrame,
    output_dir: str | Path,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Detect bubbles in every image referenced by df and 
    persist per image data.

    Parameters
    ----------
    df : pd.DataFrane
        Must contain a column image_filepath with absolute paths.
    output_dir : str | Path
        Path to save the outputs.
    debug : bool
        If True save side by side diagnostic images.

    Returns
    -------
    pandas.DataFrame
        Copy of df enriched with: 
            num_blobs_opencv number of blobs detected
            median_radii_opencv median droplet radius
    """
    if 'image_filepath' not in df.columns:
        raise ValueError("DataFrame must contain 'image_filepath' column.")
    
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    memory = Memory(location=str(out_dir / ".joblib_cache"), verbose=0)
    cached_detect = memory.cache(_detect_single_image)

    df_out = df.copy()
    num_blobs_arr = np.empty(df_out.shape[0], dtype=np.int64)
    median_r_arr = np.empty(df_out.shape[0], dtype=np.float64)

    for idx, row in tqdm(
        df_out.iterrows(),
        total=df_out.shape[0],
        desc="OpenCV SimpleBlobDetector",
    ):
        img_path = str(row.image_filepath)
        image_basename = Path(img_path).stem

        num_blobs, median_r, bubble_data = cached_detect(img_path)

        df_bubbles = pd.DataFrame(bubble_data)
        df_bubbles.to_pickle(out_dir / f"{image_basename}_bubble_data.pkl")

        if debug:
            _save_debug_overlay(img_path, bubble_data, out_dir)

        num_blobs_arr[idx] = num_blobs
        median_r_arr[idx] = median_r

    df_out["num_blobs_opencv"] = num_blobs_arr
    df_out["median_radii_opencv"] = median_r_arr
    return df_out