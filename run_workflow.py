import argparse
import logging
import yaml
from pathlib import Path
from typing import Any, Optional, Dict

from neat_ml.workflow.lib_workflow import (_as_steps_set,
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
        Comma separated list of steps. (Currently, only detect)

    Returns
    -------
    None
        Executes chosen stages in order.
    """
    steps: list[str] = _as_steps_set(steps_str)

    with open(config_path, "r") as fh:
        cfg: Any = yaml.safe_load(fh)

    roots: dict[str, Any] = cfg["roots"]
    log.info("Running steps: %s", steps)

    datasets = cfg.get("datasets", [])

    if "detect" in steps:
        log.info("\n--- STAGE: DETECT ---")
        for ds in datasets:
            paths = get_path_structure(roots, ds, steps)
            stage_detect(ds, paths)
    
    if "analysis" in steps:
        log.info("\n--- STAGE: ANALYSIS ---")
        for ds in datasets:
            paths = get_path_structure(roots, ds, steps)
            stage_analyze_features(ds, paths)

    model_path: Optional[Path] = None
    train_list = [d for d in datasets if d.get("role") == "train"]
    val_list = [d for d in datasets if d.get("role") == "val"]
    infer_list: list[Dict[str, Any]] = [d for d in datasets if d.get("role") == "infer"]

    if "train" in steps:
        if not train_list:
            log.error("No role='train' dataset.")
            return
        if len(train_list) > 1:
            log.warning("Multiple train datasets. Using the first.")
        if len(val_list) > 1:
            log.warning("Multiple val datasets. Using the first.")

        train_ds: Dict[str, Any] = train_list[0]
        val_ds: Dict[str, Any] = val_list[0]

        train_paths: Dict[str, Path] = get_path_structure(roots, train_ds, steps=["train"])
        val_paths: Optional[Dict[str, Path]] = (get_path_structure(roots, val_ds, steps=["train"]) if val_ds else None)

        model_path = stage_train_model(train_ds, train_paths, val_ds, val_paths)

    if model_path is None and any(s in steps for s in ("explain", "infer", "plot")):
        model_path_str: Optional[str] = cfg.get("inference_model")
        if not model_path_str:
            log.error("No model available. Train first or set 'inference_model' in YAML.")
            return
        model_path = Path(model_path_str).expanduser().resolve()
        if not model_path.exists():
            log.error("Model not found: %s", model_path)
            return
        log.info("Using model from config: %s", model_path)

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
            "Comma-separated list of steps to run or 'all'. Defaults to 'all'.\n"
            "Available steps: Steps: detect, analysis, train, explain, infer, plot.\n"
            "Example: --steps \"Steps: detect,analysis,train,explain,infer,plot.\""
        )
    )
    args = parser.parse_args()

    main(args.config, args.steps)

