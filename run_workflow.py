

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Set, Optional, Sequence
import yaml
from pathlib import Path
import warnings
from typing import Any, Optional

from neat_ml.workflow.lib_workflow import (as_steps_set,
                                           get_path_structure, 
                                           stage_detect,
                                           stage_analyze_features,
                                           stage_train_model,
                                           stage_run_inference_and_plot,
                                           stage_explain)

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
        str containing individual steps to perform, separated
        by commas if multiple steps provided, or 'all'.
    """
    steps = as_steps_set(steps_str)

    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    roots = cfg["roots"]
    log.info(f"Running steps: {steps}")
    
    datasets = cfg.get("datasets", [])
    if "detect" in steps:
        log.info("--- STAGE: DETECT ---")
        for ds in datasets:
            # gather `.yaml` file save path subfolders
            base_path = Path(roots.get("work"))
            dataset_id = ds.get("id")
            method = ds.get("method")
            # ``method`` is required to orchestrate downstream processess,
            # so strictly enforce user input. 
            if method is None:
                raise ValueError("Please provide `datasets:method` via input yaml file.")
            img_class = ds.get("class")
            timestamp = ds.get("time_label")
            paths = get_path_structure(roots, ds, steps)
            # run detection and return output dataframe
            df_out = stage_detect(ds, paths)
            out_path = base_path / dataset_id  / method / img_class / timestamp
            if not df_out.empty:
                df_out.to_csv(
                    out_path / "bubble_data_summary.csv"
                )
            else:
                warnings.warn("Output dataframe empty.")

    if "analysis" in steps:
        log.info("\n--- STAGE: ANALYSIS ---")
        for ds in datasets:
            paths = get_path_structure(roots, ds, steps)
            stage_analyze_features(ds, paths)
    
    model_path = roots.get("model", "")
    train_list = [d for d in datasets if d.get("role") == "train"]
    val_list = [d for d in datasets if d.get("role") == "val"]
    infer_list = [d for d in datasets if d.get("role") == "infer"]

    if "train" in steps:
        if not train_list:
            raise ValueError("No role='train' dataset.")
        if len(train_list) > 1:
            raise ValueError(
                "Multiple train datasets provided, "
                "only one can be used at a time."
            )
        if len(val_list) > 1:
            raise ValueError(
                "Multiple validation datasets provided, "
                "only one can be used at a time."
            )

        train_ds = train_list[0]
        val_ds = val_list[0]
        train_id = train_ds.get("id")
        trained_model = Path(model_path) / f"{train_id}_model.joblib"
        if not trained_model.exists():
            train_paths = get_path_structure(roots, train_ds, steps=["train"])
            val_paths = (
                get_path_structure(
                    roots, val_ds, steps=["train"]) if val_ds else None
            )
            ml_hyper_opt = train_ds.get("ml_hyper_opt", True)

            model_path = stage_train_model(
                train_ds, train_paths, val_ds, val_paths, ml_hyper_opt=ml_hyper_opt
            )
        else:
            model_path = trained_model
            log.info(f"Trained model already exists: {model_path}, skipping training...")

    if model_path == "" and any(s in steps for s in ("explain", "infer", "plot")):
        model_path_str = cfg.get("inference_model")
        if not model_path_str:
            raise ValueError("No model available. Train first or set 'inference_model' in YAML.")
        model_path = Path(model_path_str).expanduser().resolve()
        if not model_path.exists():
            raise ValueError(f"Model not found at specified path: {model_path}")
        log.info(f"Using model from config: {model_path}")

    if "explain" in steps and model_path:
        log.info("\n--- STAGE: EXPLAIN ---")
        train_ds = train_list[0] if train_list else datasets[0]
        explain_paths = get_path_structure(roots, train_ds, ["train"])
        stage_explain(train_ds, explain_paths, model_path)

    if any(s in steps for s in ("infer", "plot")) and model_path:
        log.info("\n--- STAGE: INFERENCE & PLOTTING ---")
        for ds in infer_list:
            infer_paths = get_path_structure(roots, ds, steps)
            stage_run_inference_and_plot(ds, infer_paths, model_path, steps)
    
    log.info("Workflow finished.")

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
            "Comma-separated list of steps to run or 'all'. Defaults to 'all'.\n"
            "Available steps: detect, analysis, train, explain, infer, plot.\n"
            "Example: --steps \"detect,analysis,train,explain,infer,plot.\""
        )
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(args.config, args.steps)
