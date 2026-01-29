import argparse
import logging
import yaml
from pathlib import Path
import warnings

from neat_ml.workflow.lib_workflow import (get_path_structure, 
                                           stage_detect)

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
    """
    steps = [s.strip() for s in steps_str.split(",") if s.strip()]

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
            img_class = ds.get("class")
            timestamp = ds.get("time_label")
            paths = get_path_structure(roots, ds)
            # run detection and return output dataframe
            df_out = stage_detect(ds, paths)
            out_path = base_path / dataset_id  / method / img_class / timestamp
            if df_out is not None:
                df_out.to_csv(
                    out_path / "bubble_data_summary.csv"
                )
            else:
                warnings.warn("Output dataframe empty.")

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
        default="detect",
        help=(
            "Comma-separated list of steps to run. Defaults to 'detect'.\n"
            "Available steps: detect\n"
            "Example: --steps \"detect\""
        )
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(args.config, args.steps)
