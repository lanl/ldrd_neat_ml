import logging
from pathlib import Path
from typing import Any, Optional
import pandas as pd

from neat_ml.opencv.preprocessing import process_directory as cv_preprocess
from neat_ml.opencv.detection import run_opencv
from neat_ml.bubblesam.bubblesam import run_bubblesam

__all__ = ["get_path_structure", "run_detection", "stage_detect"]

log = logging.getLogger(__name__)

def get_path_structure(
    roots: dict[str, str],
    dataset_config: dict[str, Any],
) -> dict[str, Path]:
    """
    Build only the paths needed by active steps.

    Parameters
    ----------
    roots : dict[str, str]
        Root dirs (work).
    dataset_config : dict[str, Any]
        Dataset dict (id, method, class, time_label, detection).

    Returns
    -------
    paths : dict[str, Path]
        Paths keyed by step usage (proc_dir, det_dir).
    """
    paths = {}
    ds_id = dataset_config.get("id", "unknown")
    method = dataset_config.get("method", "")
    class_label = dataset_config.get("class", "")
    time_label = dataset_config.get("time_label", "")
    work_root = Path(roots["work"])

    base_proc = work_root / ds_id / method / class_label / time_label

    if method == 'OpenCV':
        paths["proc_dir"] = base_proc / f"{time_label}_Processed_{method}"

    paths["det_dir"] = base_proc / f"{time_label}_Processed_{method}_With_Blob_Data"

    return paths

def run_detection(
    dataset_config: dict[str, Any],
    paths: dict[str, Path]
) -> Optional[pd.DataFrame]:
    """
    Run OpenCV preprocessing + detection or BubbleSAM detection when configured.

    Parameters
    ----------
    dataset_config : dict[str, Any]
        Dataset config. Expects 'method' == 'OpenCV' and 'detection' block OR
        ``method == BubbleSAM``
    paths : dict[str, Path]
        Paths from get_path_structure() (proc_dir, det_dir if built).
    
    Returns:
    --------
    df_out: Optional[pd.DataFrame]
        dataframe containing summary of bubble detection
        information
    """
    detection_cfg = dataset_config.get("detection", {})
    img_dir_str = detection_cfg.get("img_dir")
    debug = detection_cfg.get("debug", False)
    ds_id = dataset_config.get("id", "unknown")
    method = dataset_config.get("method", "")
    # get method (``opencv`` or ``bubblesam``) and initialize
    # variables to guide function calls
    if method.lower() == "opencv":
        check_dirs = set(["det_dir", "proc_dir"])
        file_suffix = "_bubble_data"
    else:
        check_dirs = set(["det_dir"])
        file_suffix = "_masks_filtered"
    
    # check if the appropriate image filepaths are available
    if not set(paths.keys()) == check_dirs:
        log.warning("Detection paths not built (step not selected or misconfig). Skipping.")
        return None
    
    # check if the input image filepaths data structure contains the appropriate
    # keys for performing detection
    if not img_dir_str:
        log.warning(f"No 'detection.img_dir' set for dataset '{ds_id}'. Skipping detection.")
        return None

    img_dir = Path(img_dir_str).expanduser().resolve()
    
    # check if the detection step has already been performed
    det_dir = paths["det_dir"].expanduser().resolve()
    det_dir.mkdir(parents=True, exist_ok=True)
    if list(det_dir.glob(f"*{file_suffix}.parquet.gzip")):
        log.info(f"Detection already exists for {ds_id}. Skipping.")
        return None
    
    # for the ``opencv`` method, perform image preprocessing
    if method.lower() == "opencv":
        proc_dir = paths["proc_dir"].expanduser().resolve()
        proc_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Preprocessing (OpenCV) for {ds_id} -> {proc_dir}")
        cv_preprocess(img_dir, proc_dir)
    else:
        proc_dir = img_dir
    
    log.info(f"Detecting ({method}) for {ds_id} -> {det_dir}")
    # collect paths for preprocessed tiff image files, store in DataFrame
    # check if the path is a single file or a directory
    if proc_dir.is_file():
        df_imgs = pd.DataFrame({"image_filepath": [proc_dir]})
    elif proc_dir.is_dir():
        img_paths = proc_dir.glob("**/*.tiff")  # type: ignore[assignment]
        df_imgs = pd.DataFrame({"image_filepath": img_paths})
    else:
        raise FileNotFoundError(
            "Invalid filepath. Must provide path to image or directory."
        )
    # run specified detection method
    if method.lower() == "opencv":
        df_out = run_opencv(df_imgs, det_dir, debug=debug)
    else:
        df_out = run_bubblesam(df_imgs, det_dir, detection_cfg=detection_cfg, debug=debug)
    log.info(f"{method} Detection Ran Successfully.")
    return df_out


def stage_detect(
    dataset_config: dict[str, Any],
    paths: dict[str, Path]
) -> pd.DataFrame:
    """
    Route detection to OpenCV or BubbleSAM based on dataset_config.method.

    Parameters
    ----------
    dataset_config : dict[str, Any]
        Dataset config with 'method'.
    paths : dict[str, Path]
        Detection paths (proc_dir, det_dir).

    Returns:
    --------
    df_out: pd.DataFrame
        dataframe containing summary of opencv bubble detection
        information 
    """
    method = dataset_config.get("method", "").lower()
    ds_id = dataset_config.get("id")
    if method in ["opencv", "bubblesam"]:
        df_out = run_detection(dataset_config, paths)
        if df_out is not None:
            return df_out
        else:
            return pd.DataFrame()
    else:
        raise ValueError(f"Unknown detection method '{method}' for dataset '{ds_id}'.")
