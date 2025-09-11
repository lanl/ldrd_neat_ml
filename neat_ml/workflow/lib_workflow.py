import logging
from pathlib import Path
from typing import Any, Optional
import pandas as pd

from neat_ml.opencv.preprocessing import process_directory as cv_preprocess
from neat_ml.opencv.detection import run_opencv
from neat_ml.bubblesam.bubblesam import run_bubblesam
from neat_ml.analysis.data_analysis import full_analysis

__all__ = ["get_path_structure", "run_detection", "stage_detect", "stage_analyze_features", "_as_steps_set"]

log = logging.getLogger(__name__)

def _as_steps_set(steps_str: str) -> list[str]:
    """
    Normalize a comma list to canonical step names.

    Parameters
    ----------
    steps_str : str
        Comma-separated steps; accepts 'detect', 'analysis'.

    Returns
    -------
    list[str]
        Normalized steps. 'all' expands to full pipeline.
    """
    raw: list[str] = [s.strip() for s in steps_str.split(",") if s.strip()]
    if raw == ["all"]:
        return ["detect", "analysis"]

    out: list[str] = []
    for s in raw:
        sl = s.lower()
        out.append(sl)
    return out

def get_path_structure(
    roots: dict[str, str],
    dataset_config: dict[str, Any],
    steps: Sequence[str]
) -> dict[str, Path]:
    """
    Build only the paths needed by active steps.

    Parameters
    ----------
    roots : dict[str, str]
        Root dirs (work).
    dataset_config : dict[str, Any]
        Dataset dict (id, method, class, time_label, detection).
    steps : Sequence[str]
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

    base_proc = work_root / ds_id / method / class_label / time_label
    results_root = Path(roots["results"])

    if method == 'OpenCV':
        paths["proc_dir"] = base_proc / f"{time_label}_Processed_{method}"

    paths["det_dir"] = base_proc / f"{time_label}_Processed_{method}_With_Blob_Data"

    if any(s in steps_set for s in {"analysis"}):
        a_cfg: Dict[str, Any] = dict(dataset_config.get("analysis", {}))
        default_per: Path = results_root / ds_id / "per_image.csv"
        default_agg: Path = results_root / ds_id / "aggregate.csv"
        paths["per_csv"] = Path(a_cfg.get("per_image_csv", default_per))
        paths["agg_csv"] = Path(a_cfg.get("aggregate_csv", default_agg))
        comp_choice: Optional[str] = a_cfg.get("composition_csv") or dataset_config.get("composition_csv")
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
       

def stage_analyze_features(dataset_config: Dict[str, Any], paths: Dict[str, Path]) -> None:
    """
    Run per-image and aggregate feature analysis for one dataset.

    Parameters
    ----------
    dataset_config : Dict[str, Any]
        Dataset config with optional 'analysis' block.
    paths : Dict[str, Path]
        Paths built for active steps.

    Returns
    -------
    None
        Writes per-image and aggregate CSVs.
    """
    ds_id: str = str(dataset_config.get("id", "unknown"))
    mode: str = str(dataset_config.get("method", ""))
    time_label: str = str(dataset_config.get("time_label", ""))

    composition_cols: list[str] = list(dataset_config.get("composition_cols", []))
    analysis_cfg: Dict[str, Any] = dict(dataset_config.get("analysis", {}))

    # input_dir: Path = Path(
    #     analysis_cfg.get("input_dir", paths.get("det_dir", ""))
    # )
    input_dir_val: Optional[str] = (
        analysis_cfg.get("input_dir")
        or (str(paths["det_dir"]) if "det_dir" in paths and paths["det_dir"] else None)
    )
    if not input_dir_val:
        log.error("No analysis input_dir provided and det_dir unavailable. Skipping '%s'.", ds_id)
        return

    input_dir: Path = Path(input_dir_val)
    per_image_csv: Path = Path(
        analysis_cfg.get("per_image_csv", paths.get("per_csv", Path()))
    )
    aggregate_csv: Path = Path(
        analysis_cfg.get("aggregate_csv", paths.get("agg_csv", Path()))
    )
    composition_csv: Optional[Path] = (
        Path(analysis_cfg["composition_csv"])
        if "composition_csv" in analysis_cfg else paths.get("composition_csv")
    )

    group_cols: list[str] = list(
        analysis_cfg.get("group_cols", ["Group", "Label", "Time", "Class"])
    )
    cols_to_add: list[str] = ["Group", "Phase_Separation"] + composition_cols
    carry_over_cols: list[str] = ["Phase_Separation"] + composition_cols

    graph_method: Optional[str] = analysis_cfg.get("graph_method",
                                                   dataset_config.get("graph_method"))
    graph_param: Optional[int | float] = analysis_cfg.get(
        "graph_param", dataset_config.get("graph_param")
    )

    if not str(input_dir):
        log.error("No analysis input_dir provided and det_dir unavailable. Skipping '%s'.",
                  ds_id)
        return
    if not input_dir.exists():
        log.error("Analysis input_dir '%s' does not exist for '%s'.", input_dir, ds_id)
        return
    if composition_csv and not Path(composition_csv).exists():
        log.error("Composition CSV '%s' missing for '%s'.", composition_csv, ds_id)
        return
    
    method_key = mode.lower()
    expected_pattern: Optional[str] = "*_bubble_data.pkl" if method_key == "opencv" else "*_masks_filtered.pkl" if method_key == "bubblesam" else None
    if expected_pattern is not None and not any(input_dir.rglob(expected_pattern)):
        log.error("No detection outputs matching '%s' under '%s' for dataset '%s' (mode='%s'). Skipping.",
                  expected_pattern, input_dir, ds_id, mode)

    aggregate_csv.parent.mkdir(parents=True, exist_ok=True)
    per_image_csv.parent.mkdir(parents=True, exist_ok=True)

    log.info("Analyzing '%s'. Input='%s' -> Per='%s', Agg='%s'.",
             ds_id, input_dir, per_image_csv, aggregate_csv)

    full_analysis(
        input_dir=input_dir,
        per_image_csv=per_image_csv,
        aggregate_csv=aggregate_csv,
        mode=mode,
        graph_method=graph_method,
        graph_param=graph_param,
        composition_csv=composition_csv,
        cols_to_add=cols_to_add,
        group_cols=group_cols,
        carry_over_cols=carry_over_cols,
        time_label=time_label,
        exclude_numeric_cols=["Offset"],
    )
