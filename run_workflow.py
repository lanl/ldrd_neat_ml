

import argparse
import logging
import yaml
from typing import Any

from neat_ml.workflow.lib_workflow import (_as_steps_set,
                                           get_path_structure, 
                                           stage_detect,
                                           stage_analyze_features)

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
            "Available steps: detect,analysis\n"
            "Example: --steps \"detect,analysis\""
        )
    )
    args = parser.parse_args()

    main(args.config, args.steps)

