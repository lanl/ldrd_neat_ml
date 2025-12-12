import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from typing import Any, Optional, Union
from pathlib import Path
import pickle
from tqdm import tqdm
import joblib
import logging

from skimage.measure import label, regionprops, find_contours
from matplotlib.patches import Rectangle

from .SAM import SAMModel

memory = joblib.Memory("joblib_cache", verbose=0)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# these settings are used to parametrize 
# ``SAM2AutomaticMaskGenerator`` and were hand-tuned
# via visual inspection to increase the number of bubbles
# detected and with consideration of computational
# cost. As noted in the `README.md`, increased speed
# and lower GPU memory can be achieved by reducing
# ``points_per_side`` from 32 to 16.
#
# NOTE: these parameters were not determined via 
# systematic hyperparameter optimization (issue #13)
DEFAULT_MASK_SETTINGS: dict[str, Any] = {
    "points_per_side": 32,
    "points_per_batch": 128,
    "pred_iou_thresh": 0.80,
    "stability_score_thresh": 0.80,
    "stability_score_offset": 0.10,
    "crop_n_layers": 4,
    "box_nms_thresh": 0.10,
    "crop_n_points_downscale_factor": 1,
    "min_mask_region_area": 5,
    "use_m2m": True,
}

DEFAULT_MODEL_CFG: dict[str, Any] = {
    "model_config": "sam2_hiera_l.yaml",
    "checkpoint_path": "./neat_ml/sam2/checkpoints/sam2_hiera_large.pt",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

def load_image(image_path: str) -> np.ndarray:
    """
    Loads and converts an image from BGR to RGB.
    
    Parameters
    ----------
    image_path : str
                 The path to the image file.
    
    Returns
    -------
    image : np.ndarray
        The loaded and converted image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_masks(masks: list[dict[str, Any]], output_path: str) -> None:
    """
    Saves the generated masks to the specified output path.
    
    Parameters
    ----------
    masks : list[dict[str, Any]]
            The generated masks.
    output_path : str
                  The path to save the masks.
    """
    for i, mask in enumerate(masks):
        mask_image = (mask['segmentation'] * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_path, f'mask_{i}.png'), mask_image)


def show_anns(anns: list[dict[str, Any]]) -> None:
    """
    Shows the mask annotations over an image.
    
    Parameters
    ----------
    anns : list[dict[str, Any]]
           The generated masks.
    """
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((
        sorted_anns[0]['segmentation'].shape[0], 
        sorted_anns[0]['segmentation'].shape[1],
        4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def analyze_and_filter_masks(
    masks_summary_df: pd.DataFrame,
    area_threshold: float = 25,
    circularity_threshold: float = 0.9
) -> pd.DataFrame:
    """
    Analyzes each mask (row) in masks_summary_df, measuring shape properties 
    and filtering out non-circular masks based on a circularity threshold.

    Parameters
    ----------
    masks_summary_df : pd.DataFrame
        DataFrame containing at least a column 'segmentation' with boolean masks.
    area_threshold : float
        Minimum area for the mask to be retained. Value of minimum area was determined
        by hand-tuning based on visual observation to exclude objects that were
        too small to be considered bubbles in the microscopy images 
    circularity_threshold : float
        Circularity threshold to consider for the mask to be "circular."
        Value for threshold was determined by hand-tuning based on visual inspection
        to exclude "debris" particles while accounting for non-perfectly circular
        shape of bubbles in microscopy images
        
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
    output_path: str
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
    output_path : str
        File path to save the resulting figure.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
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
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    torch.cuda.empty_cache()


def process_image(
    image_path: str,
    output_dir: str,
    sam_model: SAMModel,
    mask_settings: dict[str, Any],
    debug: bool = False
) -> pd.DataFrame:
    """
    Processes a single image, generates a mask, filters and analyzes it, 
    and saves debug images as needed.

    Parameters
    ----------
    image_path : str
                 The path to the input image.
    output_dir : str
                 The directory to save the masks.
    sam_model : SAMModel
                The initialized SAM model.
    mask_settings : dict[str, Any]
                    Settings for mask generation.
    debug : bool
            If True, diagnostic images (overlay and filtered contours) will be saved.

    Returns
    -------
    filtered_df : pd.DataFrame
        A DataFrame containing properties of the masks (e.g., bubbles), 
        including contour, bounding box, axes lengths, etc.
    """
    image_basename = Path(image_path).stem
    os.makedirs(output_dir, exist_ok=True)
    image = load_image(image_path)
    masks = sam_model.generate_masks(output_dir, image, mask_settings)
    masks_summary_df = sam_model.mask_summary(masks)
    
    filtered_df = analyze_and_filter_masks(
        masks_summary_df,
        area_threshold=25,
        circularity_threshold=0.90
    )

    with open(os.path.join(
        output_dir, 
        f'{image_basename}_masks_filtered.pkl'
        ), 
        'wb'
    ) as f:
        pickle.dump(filtered_df, f)

    if debug:
        plt.figure(figsize=(20,20))
        plt.imshow(image)
        show_anns(masks[1:])
        plt.axis('off')
        plt.savefig(os.path.join(
            output_dir, 
            f'{image_basename}_with_mask.png'
            )
        )
        plt.close()

        plot_filtered_masks(
            original_image=image,
            masks_summary_df=filtered_df,
            output_path=os.path.join(
                output_dir, 
                f'{image_basename}_filtered_contours.png'
            )
        )
    torch.cuda.empty_cache()

    return filtered_df

def run_bubblesam(
    df_imgs: pd.DataFrame,
    output_dir: Union[str, Path],
    *,
    model_cfg: dict[str, Any] = DEFAULT_MODEL_CFG,
    mask_settings: dict[str, Any] = DEFAULT_MASK_SETTINGS,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Primary orchestrator entry point - matches call-site in run_workflow.py.

    Parameters
    ----------
    df_imgs : pd.DataFrame
        Dataframe containing absolute image filepaths.
        Requires 'image_filepath'.
    output_dir : str | Path
        Target directory for _masks_filtered.pkl + summary CSV.
    model_cfg : dict[str, Any] | None
        Dict of settings for ``SAM2`` model. default is ``DEFAULT_MODEL_CFG``.
    mask_settings : dict[str, Any] | None
        Dict of settings for ``SAM2AutomaticMaskGenerator``.
        default is ``DEFAULT_MASK_SETTINGS``
    debug : bool
        Save graphical overlays.

    Returns
    -------
    summary : pd.DataFrame
        One row per image with detection statistics.
    """
    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    sam_model = SAMModel(**model_cfg)

    radii = np.zeros(len(df_imgs), dtype=np.float64)
    counts = np.zeros(len(df_imgs), dtype=np.int64)

    for i, img_fp in tqdm(
        enumerate(df_imgs["image_filepath"]), total=len(df_imgs), desc="[BubbleSAM]"
    ):
        stats_df = process_image(str(img_fp), str(out_dir), sam_model, mask_settings, debug)
        counts[i] = len(stats_df)
        radii[i] = np.sqrt(np.median(stats_df["area"]) / np.pi) if len(stats_df) else 0.0

    summary = df_imgs.copy()
    summary["median_radii_SAM"] = radii
    summary["num_blobs_SAM"] = counts
    summary.fillna(0, inplace=True)

    (out_dir / "bubblesam_summary.csv").write_text(summary.to_csv(index=False))
    logger.info(f"BubbleSAM processed {len(df_imgs)} images -> {out_dir}")
    return summary
