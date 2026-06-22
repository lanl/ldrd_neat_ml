import argparse
import logging
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

def main(config_path: str, steps_str: str) -> None:
    """
    Orchestrate selected workflow stages.

    Parameters
    ----------
    config_path : str
        Path to config YAML.
    steps_str : str
        Comma list of steps or 'all'.
    """
    steps = as_steps_set(steps_str)

    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    roots = cfg["roots"]
    inference_model = cfg.get("inference_model")
    log.info(f"Running steps: {steps}")
    
    datasets = cfg.get("datasets", [])
    if "detect" in steps:
        log.info("--- STAGE: DETECT ---")
        for ds in datasets:
            # gather `.yaml` file save path subfolders
            base_path = Path(roots.get("work"))
            dataset_id = ds.get("id")
            method = ds.get("method")
            img_class = ds.get("class")
            timestamp = ds.get("time_label")
            paths = get_path_structure(roots, ds)
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
    
    model_path = roots.get("model", inference_model)
    train_list = [d for d in datasets if d.get("role") == "train"]
    val_list = [d for d in datasets if d.get("role") == "val"]
    infer_list = [d for d in datasets if d.get("role") == "infer"]

    if "train" in steps:
        if not train_list:
            raise ValueError("No role='train' dataset.")
        if not val_list:
            raise ValueError("No role='validate' dataset.")
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

    if model_path is not None and any(s in steps for s in ("explain", "infer", "plot")):
        model_path = Path(model_path_str).expanduser().resolve()
        if not model_path.exists():
            raise ValueError(f"Model not found at specified path: {model_path}")
        log.info(f"Using model from config: {model_path}")
        
        if "explain" in steps:
            log.info("\n--- STAGE: EXPLAIN ---")
            train_ds = train_list[0] if train_list else datasets[0]
            explain_paths = get_path_structure(roots, train_ds, ["train"])
            stage_explain(train_ds, explain_paths, model_path)

        if any(s in steps for s in ("infer", "plot")):
            log.info("\n--- STAGE: INFERENCE & PLOTTING ---")
            for ds in infer_list:
                infer_paths = get_path_structure(roots, ds, steps)
                stage_run_inference_and_plot(ds, infer_paths, model_path, steps)
    else:
        raise ValueError("No model available. Train first or set 'inference_model' in YAML.")

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
