import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set, Sequence
import pandas as pd
from joblib import load as joblib_load

from neat_ml.opencv.preprocessing import process_directory as cv_preprocess
from neat_ml.opencv.detection import run_opencv
from neat_ml.bubblesam.bubblesam import run_bubblesam
from neat_ml.analysis.data_analysis import full_analysis
from neat_ml.model.train import (preprocess as ml_preprocess,
                                 train_with_validation, save_model_bundle,
                                 plot_roc)
from neat_ml.model.inference import run_inference
from neat_ml.model.feature_importance import compare_methods
from neat_ml.phase_diagram.plot_phase_diagram import construct_phase_diagram


__all__ = [
    "as_steps_set",
    "get_path_structure",
    "run_detection",
    "stage_detect",
    "stage_detect",
    "stage_analyze_features",
    "stage_train_model",
    "stage_run_inference_and_plot",
    "stage_explain"
]

log = logging.getLogger(__name__)

def as_steps_set(steps_str: str) -> list[str]:
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
    raw = [s.strip() for s in steps_str.split(",") if s.strip()]
    if raw == ["all"]:
        return ["detect", "analysis", "train", "infer", "explain", "plot"]

    out = []
    for s in raw:
        out.append(s.lower())
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
        Selected steps (e.g., ['detect','analysis', 'train', 'infer', 'explain', 'plot']).

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
    results_root = Path(roots["results"])
    model_root: Path = Path(roots.get("model", str(results_root / "model")))

    if method == 'OpenCV':
        paths["proc_dir"] = base_proc / f"{time_label}_Processed_{method}"

    paths["det_dir"] = base_proc / f"{time_label}_Processed_{method}_With_Blob_Data"

    if any(s in steps_set for s in {"analysis", "train", "infer", "explain"}):
        a_cfg = dataset_config.get("analysis", {})
        default_per  = results_root / ds_id / "per_image.csv"
        default_agg = results_root / ds_id / "aggregate.csv"
        paths["per_csv"] = Path(a_cfg.get("per_image_csv", default_per))
        paths["agg_csv"] = Path(a_cfg.get("aggregate_csv", default_agg))
        comp_choice = a_cfg.get("composition_csv") or dataset_config.get("composition_csv")
        if comp_choice:
            paths["composition_csv"] = Path(comp_choice)

    if any(s in steps_set for s in {"train", "infer", "explain"}):
        infer_dir: Path = results_root / f"infer_{ds_id}"
        paths["model_dir"] = model_root
        paths["explain_dir"] = results_root / ds_id / "explain"
        paths["roc_out"] = infer_dir / "roc_plots"
        paths["roc_png"] = Path("roc.png")
        paths["pred_csv"] = infer_dir / "pred.csv"
        paths["phase_dir"] = infer_dir / "phase_plots"

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
    input_dir_val = (
        analysis_cfg.get("input_dir")
        or (str(paths["det_dir"]) if "det_dir" in paths and paths["det_dir"] else None)
    )
    if not input_dir_val:
        log.warning(
            f"No analysis input_dir provided and det_dir unavailable. Skipping '{ds_id}'."
        )
        return

    # get paths for saving per image and aggregate csv files
    input_dir = Path(input_dir_val)
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
    if composition_csv and not Path(composition_csv).exists():
        log.warning(f"Composition CSV '{composition_csv}' missing for '{ds_id}'.")
        return

    group_cols = list(
        analysis_cfg.get("group_cols", ["Group", "Label", "Time", "Class"])
    )
    cols_to_add = ["Group", "Phase_Separation"] + composition_cols
    carry_over_cols = ["Phase_Separation"] + composition_cols

    graph_method = analysis_cfg.get("graph_method", dataset_config.get("graph_method"))
    graph_param = analysis_cfg.get("graph_param", dataset_config.get("graph_param"))
    
    if graph_method is None:
        raise ValueError("Please provide `graph_method` input.")
    if (graph_method.lower() in ["knn", "radius"]) and (graph_param is None):
        raise ValueError(f"Graph method: {graph_method} requires `graph_param` input.")

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
        graph_param=graph_param,
        composition_csv=composition_csv,
        cols_to_add=cols_to_add,
        group_cols=group_cols,
        carry_over_cols=carry_over_cols,
        time_label=time_label,
        exclude_numeric_cols=["Offset"],
    )


def stage_train_model(
    train_ds: Dict[str, Any],
    train_paths: Dict[str, Path],
    val_ds: Optional[Dict[str, Any]],
    val_paths: Optional[Dict[str, Path]]
) -> Path:
    """
    Train with a dedicated validation dataset and save artifacts.

    Parameters
    ----------
    train_ds : Dict[str, Any]
        Training dataset config holding 'composition_cols' etc.
    train_paths : Dict[str, Path]
        Paths for training; needs 'agg_csv' and 'model_dir'.
    val_ds : Optional[Dict[str, Any]]
        Validation dataset config used for model selection.
    val_paths : Optional[Dict[str, Path]]
        Paths for validation; needs 'agg_csv'.

    Returns
    -------
    Path
        Filesystem path to the saved model bundle (.joblib).
    """
    if val_ds is None:
        raise ValueError("stage_train_model requires a validation dataset config (val_ds).")
    if val_paths is None:
        raise ValueError("stage_train_model requires validation paths (val_paths).")
   
    ds_id: str = str(train_ds["id"])

    agg_tr: Path = train_paths["agg_csv"].expanduser().resolve()
    if not agg_tr.exists():
        raise FileNotFoundError(f"Train aggregate CSV not found: {agg_tr}")
    df_tr: pd.DataFrame = pd.read_csv(agg_tr)
    excl_tr: list[str] = ["Group", "Label", "Time", "Class"] + \
        list(train_ds.get("composition_cols", []))
    X_tr, y_tr = ml_preprocess(df_tr, target="Phase_Separation", exclude=excl_tr)

    agg_val: Path = val_paths["agg_csv"]
    if not agg_val.exists():
        raise FileNotFoundError(f"Validation aggregate CSV not found: {agg_val}")
    df_val: pd.DataFrame = pd.read_csv(agg_val)
    excl_val: list[str] = ["Group", "Label", "Time", "Class"] + \
        list(val_ds.get("composition_cols", []))
    X_val, y_val = ml_preprocess(df_val, target="Phase_Separation", exclude=excl_val)

    common_cols: list[str] = [c for c in X_tr.columns if c in X_val.columns]
    if not common_cols:
        raise ValueError("No overlapping feature columns between train and validation.")
    if len(common_cols) < len(X_tr.columns) or len(common_cols) < len(X_val.columns):
        log.warning(
            "Feature mismatch: using %d common features (train=%d, val=%d).",
            len(common_cols), len(X_tr.columns), len(X_val.columns)
        )
    X_tr = X_tr[common_cols]
    X_val = X_val[common_cols]

    model, metrics, best_params, val_proba = train_with_validation(X_tr, y_tr, X_val, y_val)

    model_dir: Path = train_paths["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path: Path = model_dir / f"{ds_id}_model.joblib"
    save_model_bundle(
        model=model,
        features=list(common_cols),
        metrics=metrics,
        best_params=best_params,
        path=str(model_path),
    )

    roc_png: Path = model_dir / f"{ds_id}_val_roc.png"
    plot_roc(y_true=y_val.to_numpy(), y_prob=val_proba, out_png=str(roc_png))
    log.info(
        "--> Model saved: %s | ROC: %s | AUC=%.3f | PR-AUC=%.3f",
        model_path,
        roc_png,
        float(metrics.get("val_roc_auc", float('nan'))),
        float(metrics.get("val_pr_auc", float('nan')))
    )
    return model_path

def stage_explain(
    train_dataset_config: Dict[str, Any],
    paths: Dict[str, Path],
    model_path: Path
) -> None:
    """
    Generates feature importance reports for a trained model.

    This function loads a pre-trained model from a .joblib file and
    the corresponding training data. It then runs various explainability
    methods (like SHAP and LIME) to determine feature importance and
    saves the resulting plots.

    Parameters
    ----------
    train_dataset_config : Dict[str, Any]
        The configuration for the training dataset, used to identify
        columns for preprocessing.
    paths : Dict[str, Path]
        A dictionary of file paths, including the training data CSV
        and the output directory for explainability plots.
    model_path : Path
        The path to the saved .joblib model bundle file.

    Returns
    -------
    None
    """
    log.info(f"--- Starting Explainability Stage for model: {model_path} ---")
    
    log.info(f"Loading training data from {paths['agg_csv']}...")
    df = pd.read_csv(paths["agg_csv"])
    
    composition_cols = train_dataset_config.get("composition_cols", [])
    exclude_cols = ["Group", "Label", "Time", "Class", "Offset"] + \
        composition_cols
    X, y = ml_preprocess(
        df, 
        target="Phase_Separation", 
        exclude=exclude_cols
    )

    log.info("Loading trained model bundle from {model_path}...")
    model_bundle = joblib_load(model_path)
    model = model_bundle['model']

    if 'features' in model_bundle and model_bundle['features'] is not None:
        log.info(f"Aligning data to the {len(model_bundle['features'])} features the model was trained on.")
        X = X[model_bundle['features']]

    log.info("Running feature importance comparison methods...")
    compare_methods(
        model=model, 
        X=X, 
        y=y, 
        out_dir=paths["explain_dir"], 
        top=20
    )
    log.info("--> Explainability plots saved to %s", paths["explain_dir"])

def stage_run_inference_and_plot(
    infer_dataset_config: Dict[str, Any],
    paths: Dict[str, Path],
    model_path: Path,
    steps: list[str]
) -> None:
    """
    Runs model inference and optionally plots 
    a phase diagram.

    Uses a trained model to make predictions on 
    an inference dataset. If specified, it then 
    uses these predictions to construct and save a
    ternary phase diagram.

    Parameters
    ----------
    infer_dataset_config : Dict[str, Any]
        The configuration for the dataset to be 
        used for inference.
    paths : Dict[str, Path]
        A dictionary of file paths for data and results.
    model_path : Path
        The path to the trained model file.
    steps : List[str]
        A list of active workflow steps to determine 
        whether to run inference, plotting, or both.

    Returns
    -------
    None
    """
    ds_id = infer_dataset_config['id']
    paths["pred_csv"].parent.mkdir(parents=True, exist_ok=True)
    composition_cols = list(infer_dataset_config.get("composition_cols", []))
    exclude_cols = ["Group", "Label", "Time", "Class", "Offset"] + composition_cols

    if "infer" in steps:
        log.info("Running inference on %s", ds_id)
        run_inference(
            model_in=model_path,
            data_csv=paths["agg_csv"],
            target="Phase_Separation",
            exclude_cols=exclude_cols,
            roc_out=paths["roc_out"],
            roc_png=paths["roc_png"],
            pred_csv=paths["pred_csv"],
        )

    if "plot" in steps:
        log.info("Constructing phase diagram for %s", ds_id)
        df_pred = pd.read_csv(paths["pred_csv"])
        print(composition_cols)
        if len(composition_cols) != 2:
            log.warning(
                "Skipping plot for %s: requires 2 composition columns.", ds_id
            )
            return
        construct_phase_diagram(
            df=df_pred,
            dex_col=composition_cols[0],
            peo_col=composition_cols[1],
            true_phase_col="Phase_Separation",
            pred_phase_col="Pred_Label",
            out_dir=paths["phase_dir"],
            title=ds_id,
            fname="phase_diagram",
        )
