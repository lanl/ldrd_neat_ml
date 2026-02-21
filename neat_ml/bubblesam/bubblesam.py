import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from typing import Any
from pathlib import Path
from tqdm import tqdm
import joblib
import logging
from matplotlib.axes import Axes
from numpy.random import Generator

from skimage.measure import label, regionprops, find_contours
from matplotlib.patches import Rectangle

from .SAM import SAMModel

memory = joblib.Memory("joblib_cache", verbose=0)

logger = logging.getLogger(__name__)


def save_masks(masks: list[dict[str, Any]], output_path: Path) -> None:
    """
    Saves the generated masks to the specified output path.
    
    Parameters
    ----------
    masks : list[dict[str, Any]]
            The generated masks.
    output_path : Path
                  The path to save the masks.
    """
    for i, mask in enumerate(masks):
        mask_image = (mask['segmentation'] * 255).astype(np.uint8)
        cv2.imwrite(
            output_path / f'mask_{i}.png',
            mask_image
        )  # type: ignore[call-overload]


def show_anns(
    anns: list[dict[str, Any]],
    ax: Axes,
    rng: Generator, 
) -> None:
    """
    Shows the mask annotations over an image.
    
    Note: this code is derived from the example jupyter notebook
    stored at:

    https://github.com/facebookresearch/sam2/blob/main/
    notebooks/automatic_mask_generator_example.ipynb

    Parameters
    ----------
    anns : list[dict[str, Any]]
           The generated masks.
    ax : Axes
        matplotlib figure axes
    rng : Generator
        psuedorandom number generator
    """
    if len(anns) == 0:
        return
    # sort the annotations from largest to smallest so that
    # large areas do not cover small areas
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax.set_autoscale_on(False)

    img = np.ones((
        sorted_anns[0]['segmentation'].shape[0], 
        sorted_anns[0]['segmentation'].shape[1],
        4))
    img[:,:,3] = 0

    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([rng.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def analyze_and_filter_masks(
    masks_summary_df: pd.DataFrame,
    area_threshold: float,
    circularity_threshold: float
) -> pd.DataFrame:
    """
    Analyzes each mask (row) in masks_summary_df, measuring shape properties 
    and filtering out non-circular masks based on a circularity threshold.

    Parameters
    ----------
    masks_summary_df : pd.DataFrame
        DataFrame containing at least a column 'segmentation' with boolean masks.
    area_threshold : float
        Minimum area for the mask to be retained. Default value of minimum area
        was determined by hand-tuning based on visual observation to exclude objects
        that were too small to be considered bubbles in the microscopy images 
    circularity_threshold : float
        Circularity threshold to consider for the mask to be "circular."
        Default value for threshold was determined by hand-tuning based on visual
        inspection to exclude "debris" particles while accounting for non-perfectly
        circular shape of bubbles in microscopy images
        
    Returns
    -------
    df_filtered : pd.DataFrame
        A filtered DataFrame with new columns (contour, bbox, major_axis, minor_axis, 
        area, radius, circ, euler_number) but excluding 'segmentation'.
    """
    filtered_rows = []

    for idx, row in masks_summary_df.iterrows():
        seg = row['segmentation']
        labeled_seg = label(seg)
        props_list = regionprops(labeled_seg)
        if len(props_list) == 0:
            continue

        rp = props_list[0]
        area = rp.area
        perimeter = rp.perimeter
        if perimeter == 0:
            continue

        circ = (4.0 * np.pi * area) / (perimeter ** 2)
        major_axis = rp.major_axis_length
        minor_axis = rp.minor_axis_length
        if area >= area_threshold and circ >= circularity_threshold:
            mask_info = {              
                'bbox': rp.bbox,
                'contour': find_contours(seg, level=0.5)[0],
                'major_axis': major_axis,
                'minor_axis': minor_axis,
                'area': area,
                'radius': np.sqrt(area / np.pi),
                'circ': circ,
                'euler_number': rp.euler_number
            }

            filtered_rows.append(mask_info)

    df_filtered = pd.DataFrame(filtered_rows)
    return df_filtered

def plot_filtered_masks(
    original_image: np.ndarray,
    masks_summary_df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Plots the filtered masks (contours + bounding boxes) on the original image 
    and saves the resulting figure.

    Parameters
    ----------
    original_image : np.ndarray
        Original image array.
    masks_summary_df : pd.DataFrame
        DataFrame containing the columns 'contour' and 'bbox' for each mask.
    output_path : Path
        File path to save the resulting figure.
    """
    fig, ax = plt.subplots(1, 1)
    ax.imshow(original_image, cmap="gray")

    for idx, row in masks_summary_df.iterrows():
        contour = row['contour']
        bbox = row['bbox']
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='blue')
        min_row, min_col, max_row, max_col = bbox
        rect = Rectangle(
            (min_col, min_row),
            max_col - min_col,
            max_row - min_row,
            linewidth=1,
            edgecolor='green',
            facecolor='none'
        )
        ax.add_patch(rect)
    ax.axis("off")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def bubblesam_detection(
    image_path: Path,
    output_dir: Path,
    sam_model: SAMModel,
    rng: Generator,
    area_threshold: float,
    circularity_threshold: float,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Processes a single image, generates a mask, filters and analyzes it, 
    and saves debug images as needed.

    Parameters
    ----------
    image_path : Path
                 The path to the input image.
    output_dir : Path
                 The directory to save the masks.
    sam_model : SAMModel
                The initialized SAM model.
    rng : Generator
          pseudorandom number generator
    area_threshold: float
        threshold for minimum area to count detection as a bubble
    circularity_threshold: float
        threshold for circularity of detection to count as a bubble
    debug : bool
            If True, diagnostic images (overlay and filtered contours) will be saved.

    Returns
    -------
    filtered_df : pd.DataFrame
        A DataFrame containing properties of the masks (e.g., bubbles), 
        including contour, bounding box, axes lengths, etc.
    """
    image_basename = image_path.stem
    image = cv2.imread(image_path)  # type: ignore[call-overload]
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    masks = sam_model.generate_masks(image)
    masks_summary_df = pd.DataFrame(masks)
    
    filtered_df = analyze_and_filter_masks(
        masks_summary_df,
        area_threshold,
        circularity_threshold
    )
   
    # save filtered dataframe as parquet file
    # convert ``contours`` column to list to save as parquet
    save_filtered_df = filtered_df.copy()
    save_filtered_df['contour'] = save_filtered_df['contour'].apply(
        lambda x: x.tolist() if isinstance(x, np.ndarray) else x
    )
    save_filtered_df.to_parquet(
        output_dir / f'{image_basename}_masks_filtered.parquet.gzip'
    )

    if debug:
        fig, ax = plt.subplots(1,1)
        ax.imshow(image)
        show_anns(masks[1:], ax, rng)
        ax.axis('off')
        fig.savefig(
            output_dir / f'{image_basename}_with_mask.png',
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        plot_filtered_masks(
            original_image=image,
            masks_summary_df=filtered_df,
            output_path=output_dir / f'{image_basename}_filtered_contours.png'
        )
    if (torch.cuda.is_available() and sam_model.device != "cpu"):
        torch.cuda.empty_cache()
    elif (torch.backends.mps.is_available() and sam_model.device != "cpu"):
        torch.mps.empty_cache()

    return filtered_df

def run_bubblesam(
    df_imgs: pd.DataFrame,
    output_dir: Path,
    *,
    model_cfg: dict[str, Any],
    debug: bool = False,
) -> pd.DataFrame:
    """
    Perform detection of bubbles in input images using SAM2

    Parameters
    ----------
    df_imgs : pd.DataFrame
        Dataframe containing absolute image filepaths.
        Requires column name: 'image_filepath'.
    output_dir : Path
        Target directory for _masks_filtered parquet + summary CSV.
    model_cfg : dict[str, Any]
        Dict of settings for ``SAM2`` model.
    debug : bool
        Save graphical overlays.

    Returns
    -------
    summary : pd.DataFrame
        One row per image with detection statistics.
    """
    # initialize a pseudorandom number generator
    rng = np.random.default_rng(seed=0)
    
    out_dir = output_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    sam_model = SAMModel(**model_cfg)

    radii = np.zeros(len(df_imgs), dtype=np.float64)
    counts = np.zeros(len(df_imgs), dtype=np.int64)

    # get input threshold values or else use defaults.
    area_threshold = model_cfg.get("area_threshold", 25.0)
    circularity_threshold = model_cfg.get("circularity_threshold", 0.90)
    for i, img_fp in tqdm(
        enumerate(df_imgs["image_filepath"]), total=len(df_imgs), desc="[BubbleSAM]"
    ):
        stats_df = bubblesam_detection(
            img_fp,
            out_dir,
            sam_model,
            rng,
            area_threshold,
            circularity_threshold,
            debug,
        )
        counts[i] = len(stats_df)
        radii[i] = np.sqrt(np.median(stats_df["area"]) / np.pi) if len(stats_df) else 0.0

    summary = df_imgs.copy()
    summary["median_radii_SAM"] = radii
    summary["num_blobs_SAM"] = counts
    summary.fillna(0, inplace=True)

    summary.to_csv(out_dir / "bubblesam_summary.csv", index=False)
    logger.info(f"BubbleSAM processed {len(df_imgs)} images -> {out_dir}")
    return summary
