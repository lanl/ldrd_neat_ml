import logging
from pathlib import Path
from typing import Any, Optional
import pandas as pd

from neat_ml.opencv.preprocessing import process_directory as cv_preprocess
from neat_ml.opencv.detection import collect_tiff_paths, run_opencv

__all__ = ["get_path_structure", "stage_opencv", "stage_detect"]

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
    dict[str, Path]
        Paths keyed by step usage (proc_dir, det_dir).
    """
    paths: dict[str, Path] = {}

    ds_id: str = str(dataset_config.get("id", "unknown"))
    method: str = str(dataset_config.get("method", ""))
    class_label: str = str(dataset_config.get("class", ""))
    time_label: str = str(dataset_config.get("time_label", ""))
    work_root: Path = Path(roots["work"])

    base_proc: Path = work_root / ds_id / method / class_label / time_label
    paths["proc_dir"] = base_proc / f"{time_label}_Processed_{method}"
    paths["det_dir"] = base_proc / f"{time_label}_Processed_{method}_With_Blob_Data"

    return paths

def stage_opencv(dataset_config: dict[str, Any], paths: dict[str, Path]) -> None:
    """
    Run OpenCV preprocessing + detection when configured.

    Parameters
    ----------
    dataset_config : dict[str, Any]
        Dataset config. Expects 'method' == 'OpenCV' and 'detection' block.
    paths : ict[str, Path]
        Paths from get_path_structure() (proc_dir, det_dir if built).

    Returns
    -------
    None
        Writes preprocessed images and detection outputs if configured.
    """
    detection_cfg: dict[str, Any] = dict(dataset_config.get("detection", {}))
    img_dir_str: Optional[str] = detection_cfg.get("img_dir")
    debug: bool = bool(detection_cfg.get("debug", False))

    if "proc_dir" not in paths or "det_dir" not in paths:
        log.warning("Detection paths not built (step not selected or misconfig). Skipping.")
        return
    if not img_dir_str:
        log.warning("No 'detection.img_dir' set for dataset '%s'. Skipping detection.",
                    dataset_config.get("id"))
        return

    proc_dir: Path = paths["proc_dir"]
    det_dir: Path = paths["det_dir"]
    img_dir: Path = Path(img_dir_str)

    ds_id: str = str(dataset_config.get("id", "unknown"))
    if list(det_dir.glob("*_bubble_data.pkl")):
        log.info("Detection already exists for %s. Skipping.", ds_id)
        return

    proc_dir.mkdir(parents=True, exist_ok=True)
    det_dir.mkdir(parents=True, exist_ok=True)

    log.info("Preprocessing (OpenCV) for %s -> %s", ds_id, proc_dir)
    cv_preprocess(img_dir, proc_dir)

    log.info("Detecting (OpenCV) for %s -> %s", ds_id, det_dir)
    img_paths = collect_tiff_paths(proc_dir)
    df_imgs = pd.DataFrame({"image_filepath": img_paths})
    run_opencv(df_imgs, det_dir, debug=debug)

def stage_detect(dataset_config: dict[str, Any], paths: dict[str, Path]) -> None:
    """
    Route detection to OpenCV based on dataset.method.

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
    if method == "opencv":
        stage_opencv(dataset_config, paths)
    else:
        log.warning("Unknown detection method '%s' for dataset '%s'.",
                    method, dataset_config.get("id"))
