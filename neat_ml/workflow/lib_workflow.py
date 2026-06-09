import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Literal
import pandas as pd

from neat_ml.opencv.preprocessing import process_directory as cv_preprocess
from neat_ml.opencv.detection import run_opencv
from neat_ml.bubblesam.bubblesam import run_bubblesam
from neat_ml.analysis.data_analysis import full_analysis


__all__ = [
    "as_steps_set",
    "get_path_structure",
    "run_detection",
    "stage_detect",
    "stage_analyze_features"
]

log = logging.getLogger(__name__)

def as_steps_set(
    steps_str: Literal["detect", "analysis", "detect,analysis", "all"]
) -> list[str]:
    """
    Normalize a comma separated string of steps
    to a list of canonical step names.

    Parameters
    ----------
    steps_str : str
        Comma-separated steps; accepts 'detect', 'analysis', 'all'.
        'all' expands to full pipeline.

    Returns
    -------
    list[str]
        List of normalized steps.
    """
    raw = [s.strip().lower() for s in steps_str.split(",") if s.strip()]
    if raw == ["all"]:
        return ["detect", "analysis"]

    return raw

def get_path_structure(
    roots: dict[str, str],
    dataset_config: dict[str, Any],
    steps: Sequence[Literal["detect", "analysis"]]
) -> dict[str, Path]:
    """
    Build only the paths needed by active steps.

    Parameters
    ----------
    roots : dict[str, str]
        Root dirs (work).
    dataset_config : dict[str, Any]
        Dataset dict (id, method, class, time_label, detection).
    steps : Sequence[Literal["detect", "analysis"]]
        Selected steps (e.g., ['detect','analysis']).

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
    steps_set = set(steps)

    base_proc = work_root / ds_id / method / class_label / time_label

    if method == 'OpenCV':
        paths["proc_dir"] = base_proc / f"{time_label}_Processed_{method}"

    paths["det_dir"] = base_proc / f"{time_label}_Processed_{method}_With_Blob_Data"

    if "analysis" in steps_set:
        results_root = Path(roots["results"])
        a_cfg = dataset_config.get("analysis", {})
        default_per  = results_root / ds_id / "per_image.csv"
        default_agg = results_root / ds_id / "aggregate.csv"
        paths["per_csv"] = Path(a_cfg.get("per_image_csv", default_per))
        paths["agg_csv"] = Path(a_cfg.get("aggregate_csv", default_agg))
        comp_choice = a_cfg.get("composition_csv") or dataset_config.get("composition_csv")
        if comp_choice:
            paths["composition_csv"] = Path(comp_choice)

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
        information OR empty dataframe that propagates through
        `run_workflow.py` if dataset errors are raised in
        `run_detection`.
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
        
def stage_analyze_features(dataset_config: dict[str, Any], paths: dict[str, Path]) -> None:
    """
    Run per-image and aggregate feature analysis for one dataset.

    Parameters
    ----------
    dataset_config : dict[str, Any]
        Dataset config with optional 'analysis' block.
    paths : dict[str, Path]
        Paths built for active steps.
    """
    # gather dataset configuration settings
    ds_id = dataset_config.get("id", "unknown")
    mode = dataset_config.get("method", "")
    time_label = dataset_config.get("time_label", "")

    composition_cols = dataset_config.get("composition_cols", [])
    analysis_cfg = dataset_config.get("analysis", {})

    # get the user provided input path storing parquet files OR
    # the detection dir where parquets were saved after detection
    analysis_input_dir = analysis_cfg.get("input_dir")
    input_dir = (
        Path(analysis_input_dir) if analysis_input_dir 
        else (paths["det_dir"] if "det_dir" in paths and paths["det_dir"] else None)
    )
    if not input_dir:
        log.warning(
            f"No analysis input_dir provided and det_dir unavailable. Skipping '{ds_id}'."
        )
        return

    # get paths for saving per image and aggregate csv files
    if not input_dir.exists():
        log.warning(f"Analysis input_dir '{input_dir}' does not exist for '{ds_id}'.")
        return

    per_image_csv = Path(
        analysis_cfg.get("per_image_csv") or paths.get("per_csv") or Path.cwd()
    )
    aggregate_csv = Path(
        analysis_cfg.get("aggregate_csv") or paths.get("agg_csv") or Path.cwd()
    )
    # get the path for the composition csv from input configuration
    composition_csv = (
        Path(analysis_cfg["composition_csv"])
        if "composition_csv" in analysis_cfg else paths.get("composition_csv")
    )
    if composition_csv and not composition_csv.exists():
        log.warning(f"Composition CSV '{composition_csv}' missing for '{ds_id}'.")
        return

    group_cols = analysis_cfg.get("group_cols", ["Group", "Label", "Time", "Class"])
    cols_to_add = ["Group", "Phase_Separation"] + composition_cols
    carry_over_cols = ["Phase_Separation"] + composition_cols

    graph_method = analysis_cfg.get("graph_method", dataset_config.get("graph_method"))
    k_param = analysis_cfg.get("k_param", dataset_config.get("k_param"))
    r_param = analysis_cfg.get("r_param", dataset_config.get("r_param"))
    
    if graph_method is None:
        raise ValueError("Please provide `graph_method` input.")
    if ((graph_method.lower() == "knn" and k_param is None)
        or (graph_method.lower() == "radius" and r_param is None)):
        raise ValueError(
            (f"Graph method: {graph_method} requires appropriate"
            "param input (i.e. `k_param` or `r_param`).")
        )

    method_key = mode.lower()
    expected_pattern = ("*_bubble_data.parquet.gzip" if method_key == "opencv"
        else "*_masks_filtered.parquet.gzip" if method_key == "bubblesam" else None
    )
    if expected_pattern is not None and not any(input_dir.rglob(expected_pattern)):
        log.warning(
            (f"No detection outputs matching '{expected_pattern}' under"
            f"'{input_dir}' for dataset '{ds_id}' (mode='{mode}'). Skipping.")
        )
        return

    aggregate_csv.parent.mkdir(parents=True, exist_ok=True)
    per_image_csv.parent.mkdir(parents=True, exist_ok=True)

    log.info(
        (
            f"Analyzing '{ds_id}'. Input='{input_dir}' ->"
            f"Per='{per_image_csv}', Agg='{aggregate_csv}'."
        )
    )
    
    full_analysis(
        input_dir=input_dir,
        per_image_csv=per_image_csv,
        aggregate_csv=aggregate_csv,
        mode=mode,
        graph_method=graph_method,
        r_param=r_param,
        k_param=k_param,
        composition_csv=composition_csv,
        cols_to_add=cols_to_add,
        group_cols=group_cols,
        carry_over_cols=carry_over_cols,
        time_label=time_label,
        exclude_numeric_cols=["Offset"],
    )
