import logging
from pathlib import Path
from typing import Any, Optional

from neat_ml.opencv.preprocessing import process_directory as cv_preprocess
from neat_ml.opencv.detection import (build_df_from_img_paths,
                                      collect_tiff_paths, run_opencv)
from neat_ml.bubblesam.bubblesam import run_bubblesam

__all__ = ["get_path_structure", "stage_opencv", "stage_bubblesam", "stage_detect"]

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
        Dataset dict (id, method, class, time_label, detection, analysis).

    Returns
    -------
    dict[str, Path]
        Paths keyed by step usage (det_dir, per_csv).
    """
    paths: dict[str, Path] = {}

    ds_id: str = str(dataset_config.get("id", "unknown"))
    method: str = str(dataset_config.get("method", ""))
    class_label: str = str(dataset_config.get("class", ""))
    time_label: str = str(dataset_config.get("time_label", ""))

    work_root: Path = Path(roots["work"])

    base_proc: Path = work_root / ds_id / method / class_label / time_label

    if method == 'OpenCV':
        paths["proc_dir"] = base_proc / f"{time_label}_Processed_{method}"

    paths["det_dir"] = base_proc / f"{time_label}_Processed_{method}_With_Blob_Data"

    return paths

def run_detection(
    dataset_config: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    """
    Run OpenCV preprocessing + detection or BubbleSAM detection when configured.

    Parameters
    ----------
    dataset_config : dict[str, Any]
        Dataset config. Expects 'method' == 'OpenCV' and 'detection' block OR
        ``method == BubbleSAM``
    paths : dict[str, Path]
        Paths from get_path_structure() (proc_dir, det_dir if built).
    """
    # get method (``opencv`` or ``bubblesam``) and initialize
    # variables to guide function calls
    method = dataset_config.get("method")
    if method.lower() == "opencv":
        check_dirs = set(["det_dir", "proc_dir"])
        file_suffix = "_bubble_data"
    else:
        check_dirs = set(["det_dir"])
        file_suffix = "_masks_filtered"
    
    # check if the appropriate image filepaths are available
    if not set(paths.keys()) == check_dirs:
        log.warning("Detection paths not built (step not selected or misconfig). Skipping.")
        return
    
    # check if the input image filepaths data structure contains the appropriate
    # keys for performing detection
    det_dir: Path = paths["det_dir"]
    detection_cfg: dict[str, Any] = dict(dataset_config.get("detection", {}))
    img_dir_str: Optional[str] = detection_cfg.get("img_dir", dataset_config.get("img_dir"))
    if not img_dir_str:
        log.warning("No 'detection.img_dir' set for dataset '%s'. Skipping detection.",
                    dataset_config.get("id"))
        return
    
    # check if the detection step has already been performed
    img_dir: Path = Path(img_dir_str)
    det_dir.mkdir(parents=True, exist_ok=True)
    ds_id: str = str(dataset_config.get("id", "unknown"))
    if list(det_dir.glob(f"*{file_suffix}.pkl")):
        log.info("Detection already exists for %s. Skipping.", ds_id)
        return
    
    # for the ``opencv`` method, perform image preprocessing
    if method.lower() == "opencv":
        debug: bool = bool(detection_cfg.get("debug", False))
        tiff_paths: Path = paths["proc_dir"]
        tiff_paths.mkdir(parents=True, exist_ok=True)
        log.info("Preprocessing (OpenCV) for %s -> %s", ds_id, tiff_paths)
        cv_preprocess(img_dir, tiff_paths)
    else:
        tiff_paths = img_dir
    
    # route the detection step to the appropriate method
    log.info(f"Detecting ({method}) for %s -> %s", ds_id, det_dir)
    img_paths = collect_tiff_paths(tiff_paths)
    df_imgs = build_df_from_img_paths(img_paths)
    if method.lower() == "opencv":
        run_opencv(df_imgs, det_dir, debug=debug)
    else:
        run_bubblesam(df_imgs, det_dir)


def stage_detect(dataset_config: dict[str, Any], paths: dict[str, Path]) -> None:
    """
    Route detection to OpenCV or BubbleSAM based on dataset.method.

    Parameters
    ----------
    dataset_config : dict[str, Any]
        Dataset config with 'method'.
    paths : dict[str, Path]
        Detection paths (proc_dir, det_dir).

    Returns
    -------
    None
        Runs the appropriate detection stage or logs a warning.
    """
    method: str = str(dataset_config.get("method", "")).lower()
    if method in ["opencv", "bubblesam"]:
        run_detection(dataset_config, paths)
    else:
        log.warning("Unknown detection method '%s' for dataset '%s'.",
                    method, dataset_config.get("id"))
