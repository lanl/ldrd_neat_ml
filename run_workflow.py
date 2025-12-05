import argparse
import logging
import yaml

from typing import Any

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
        cfg: Any = yaml.safe_load(fh)

    roots: dict[str, Any] = cfg["roots"]
    log.info("Running steps: %s", steps)

    datasets = cfg.get("datasets", [])

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
        default="detect",
        help=(
            "Comma-separated list of steps to run. Defaults to 'detect'.\n"
            "Available steps: detect\n"
            "Example: --steps \"detect\""
        )
    )
    args = parser.parse_args()

    main(args.config, args.steps)
