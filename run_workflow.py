import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Set, Optional, Sequence
import yaml

from neat_ml.opencv.preprocessing import \
    process_directory as cv_preprocess
from neat_ml.opencv.detection import (build_df_from_img_paths,
                                      collect_tiff_paths, run_opencv)
from neat_ml.bubblesam.bubblesam import run_bubblesam

log = logging.getLogger(__name__)

def _as_steps_set(steps_str: str) -> list[str]:
    """
    Normalize a comma list to canonical step names.

    Parameters
    ----------
    steps_str : str
        Comma-separated steps; accepts 'detect'.

    Returns
    -------
    list[str]
        Normalized steps. 'all' expands to full pipeline.
    """
    raw: list[str] = [s.strip() for s in steps_str.split(",") if s.strip()]
    if raw == ["all"]:
        return ["detect"]
    else:
        return raw

def get_path_structure(
    roots: Dict[str, str],
    dataset_config: Dict[str, Any],
    steps: Sequence[str]
) -> Dict[str, Path]:
    """
    Build only the paths needed by active steps.

    Parameters
    ----------
    roots : Dict[str, str]
        Root dirs (work).
    dataset_config : Dict[str, Any]
        Dataset dict (id, method, class, time_label, detection, analysis).
    steps : Sequence[str]
        Selected steps (e.g., ['detect']).

    Returns
    -------
    Dict[str, Path]
        Paths keyed by step usage (det_dir, per_csv).
    """
    paths: Dict[str, Path] = {}
    steps_set: Set[str] = set(steps)

    ds_id: str = str(dataset_config.get("id", "unknown"))
    method: str = str(dataset_config.get("method", ""))
    class_label: str = str(dataset_config.get("class", ""))
    time_label: str = str(dataset_config.get("time_label", ""))

    work_root: Path = Path(roots["work"])

    if "detect" in steps_set:
        base_proc: Path = work_root / ds_id / method / class_label / time_label
        paths["proc_dir"] = base_proc / f"{time_label}_Processed_{method}"
        paths["det_dir"] = base_proc / f"{time_label}_Processed_{method}_With_Blob_Data"

    return paths

def stage_opencv(dataset_config: Dict[str, Any], paths: Dict[str, Path]) -> None:
    """
    Run OpenCV preprocessing + detection when configured.

    Parameters
    ----------
    dataset_config : Dict[str, Any]
        Dataset config. Expects 'method' == 'OpenCV' and 'detection' block.
    paths : Dict[str, Path]
        Paths from get_path_structure() (proc_dir, det_dir if built).

    Returns
    -------
    None
        Writes preprocessed images and detection outputs if configured.
    """
    method: str = str(dataset_config.get("method", ""))
    if method != "OpenCV":
        log.info("Skipping OpenCV stage: method='%s'.", method)
        return

    detection_cfg: Dict[str, Any] = dict(dataset_config.get("detection", {}))
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
    df_imgs = build_df_from_img_paths(img_paths)
    run_opencv(df_imgs, det_dir, debug=debug)

def stage_bubblesam(dataset_config: Dict[str, Any], paths: Dict[str, Path]) -> None:
    """
    Run BubbleSAM detection when method='BubbleSAM'.

    Parameters
    ----------
    dataset_config : Dict[str, Any]
        Dataset config. Expects method 'BubbleSAM'.
        Uses detection.img_dir (falls back to dataset.img_dir).
    paths : Dict[str, Path]
        Must include proc_dir and det_dir.

    Returns
    -------
    None
        Writes preprocessed images and *_masks_filtered.pkl.
    """
    if "proc_dir" not in paths or "det_dir" not in paths:
        log.warning("Missing detection paths (not selected or misconfigured). Skipping.")
        return

    det_cfg: Dict[str, Any] = dict(dataset_config.get("detection", {}))
    img_dir_str: Optional[str] = det_cfg.get("img_dir", dataset_config.get("img_dir"))
    if not img_dir_str:
        log.warning("No detection.img_dir set for dataset '%s'. Skipping.", dataset_config.get("id"))
        return

    ds_id: str = str(dataset_config.get("id", "unknown"))
    proc_dir: Path = paths["proc_dir"]
    det_dir: Path = paths["det_dir"]
    img_dir: Path = Path(img_dir_str)

    if list(det_dir.glob("*_masks_filtered.pkl")):
        log.info("BubbleSAM outputs exist for %s. Skipping.", ds_id)
        return

    proc_dir.mkdir(parents=True, exist_ok=True)
    det_dir.mkdir(parents=True, exist_ok=True)

    log.info("Preprocessing (BubbleSAM) for %s -> %s", ds_id, proc_dir)
    cv_preprocess(img_dir, proc_dir)

    log.info("Detecting (BubbleSAM) for %s -> %s", ds_id, det_dir)
    img_paths = collect_tiff_paths(proc_dir)
    df_imgs = build_df_from_img_paths(img_paths)
    run_bubblesam(df_imgs, det_dir)

def stage_detect(dataset_config: Dict[str, Any], paths: Dict[str, Path]) -> None:
    """
    Route detection to OpenCV or BubbleSAM based on dataset.method.

    Parameters
    ----------
    dataset_config : Dict[str, Any]
        Dataset config with 'method'.
    paths : Dict[str, Path]
        Detection paths (proc_dir, det_dir).

    Returns
    -------
    None
        Runs the appropriate detection stage or logs a warning.
    """
    method: str = str(dataset_config.get("method", "")).lower()
    if method == "opencv":
        stage_opencv(dataset_config, paths)
    elif method == "bubblesam":
        stage_bubblesam(dataset_config, paths)
    else:
        log.warning("Unknown detection method '%s' for dataset '%s'.",
                    method, dataset_config.get("id"))

def main(config_path: str, steps_str: str) -> None:
    """
    Orchestrate selected workflow stages.

    Parameters
    ----------
    config_path : str
        Path to config YAML.
    steps_str : str
        Comma list of steps or 'all'.

    Returns
    -------
    None
        Executes chosen stages in order.
    """
    steps: list[str] = _as_steps_set(steps_str)

    with open(config_path, "r") as fh:
        cfg: Any = yaml.safe_load(fh)

    roots: Dict[str, str] = cfg["roots"]
    log.info("Running steps: %s", steps)

    datasets: list[Dict[str, Any]] = cfg.get("datasets", [])
    
    if "detect" in steps:
        log.info("\n--- STAGE: DETECT ---")
        for ds in datasets:
            paths = get_path_structure(roots, ds)
            stage_detect(ds, paths)

    log.info("\nWorkflow finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full or partial NEAT-ML workflow.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        '--steps',
        type=str,
        default="all",
        help=(
            "Comma-separated list of steps to run. Defaults to 'all'.\n"
            "Available steps: detect\n"
            "Example: --steps \"detect\""
        )
    )
    args = parser.parse_args()

    main(args.config, args.steps)
