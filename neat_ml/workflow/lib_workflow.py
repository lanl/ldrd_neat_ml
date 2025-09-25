import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set, Sequence

from neat_ml.opencv.preprocessing import process_directory as cv_preprocess
from neat_ml.opencv.detection import (build_df_from_img_paths,
                                      collect_tiff_paths, run_opencv)
from neat_ml.bubblesam.bubblesam import run_bubblesam
from neat_ml.analysis.data_analysis import full_analysis

__all__ = ["_as_steps_set", "get_path_structure", "stage_opencv", "stage_bubblesam", "stage_detect", "stage_analyze_features"]

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
        Selected steps (e.g., ['detect','analysis']).

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
    results_root: Path = Path(roots["results"])
    base_proc: Path = work_root / ds_id / method / class_label / time_label

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
    if "det_dir" not in paths:
        log.warning("Missing detection paths (not selected or misconfigured). Skipping.")
        return

    det_cfg: Dict[str, Any] = dict(dataset_config.get("detection", {}))
    img_dir_str: Optional[str] = det_cfg.get("img_dir", dataset_config.get("img_dir"))
    if not img_dir_str:
        log.warning("No detection.img_dir set for dataset '%s'. Skipping.", dataset_config.get("id"))
        return

    ds_id: str = str(dataset_config.get("id", "unknown"))
    det_dir: Path = paths["det_dir"]
    img_dir: Path = Path(img_dir_str)

    if list(det_dir.glob("*_masks_filtered.pkl")):
        log.info("BubbleSAM outputs exist for %s. Skipping.", ds_id)
        return

    det_dir.mkdir(parents=True, exist_ok=True)
    log.info("Detecting (BubbleSAM) for %s -> %s", ds_id, det_dir)
    img_paths = collect_tiff_paths(img_dir)
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