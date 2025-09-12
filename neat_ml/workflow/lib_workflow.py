import pandas as pd
import logging
from pathlib import Path
from joblib import load as joblib_load
from typing import Any, Dict, Optional, Set, Sequence

from neat_ml.opencv.preprocessing import process_directory as cv_preprocess
from neat_ml.opencv.detection import (build_df_from_img_paths,
                                      collect_tiff_paths, run_opencv)
from neat_ml.bubblesam.bubblesam import run_bubblesam
from neat_ml.analysis.data_analysis import full_analysis
from neat_ml.model.train import (preprocess as ml_preprocess,
                                 train_with_validation, save_model_bundle,
                                 plot_roc)
from neat_ml.model.inference import run_inference
from neat_ml.model.feature_importance import compare_methods
from neat_ml.phase_diagram.plot_phase_diagram import construct_phase_diagram

__all__ = ["_as_steps_set", 
           "get_path_structure", 
           "stage_opencv", 
           "stage_bubblesam",
           "stage_detect", 
           "stage_analyze_features",
           "stage_train_model",
           "stage_run_inference_and_plot",
           "stage_explain"
           ]

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
        return ["detect", "analysis", "train", "infer", "explain", "plot"]

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
        Selected steps (e.g., ['detect','analysis', 'train', 'infer', 'explain', 'plot']).

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
    model_root: Path = Path(roots.get("model", str(results_root / "model")))
    base_proc: Path = work_root / ds_id / method / class_label / time_label

    if method == 'OpenCV':
        paths["proc_dir"] = base_proc / f"{time_label}_Processed_{method}"

    paths["det_dir"] = base_proc / f"{time_label}_Processed_{method}_With_Blob_Data"

    if any(s in steps_set for s in {"analysis", "train", "infer", "explain"}):
        a_cfg: Dict[str, Any] = dict(dataset_config.get("analysis", {}))
        default_per: Path = results_root / ds_id / "per_image.csv"
        default_agg: Path = results_root / ds_id / "aggregate.csv"
        paths["per_csv"] = Path(a_cfg.get("per_image_csv", default_per))
        paths["agg_csv"] = Path(a_cfg.get("aggregate_csv", default_agg))
        comp_choice: Optional[str] = a_cfg.get("composition_csv") or dataset_config.get("composition_csv")
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