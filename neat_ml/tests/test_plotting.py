from __future__ import annotations

import json
from pathlib import Path
from typing import Callable
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.testing.compare import compare_images

from neat_ml import plot_manuscript_figures as pmf
from neat_ml import figure_utils

@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """Deterministic (seeded) composition/phase dataframe."""
    rng = np.random.default_rng(7)
    x = rng.uniform(0.0, 20.0, 30)
    y = rng.uniform(0.0, 20.0, 30)
    phase = (x + y > 20.0).astype(int)
    return pd.DataFrame({"X": x, "Y": y, "Phase": phase})


@pytest.fixture(scope="session")
def baseline_dir() -> Path:
    """Directory that stores the reference (golden) images."""
    return Path(__file__).parent / "baseline"

pytestmark = pytest.mark.usefixtures("baseline_dir", "synthetic_df")

def assert_same_image(result: Path, baseline: Path, *, tol: float = 10.0):
    """
    Fail if the two PNGs differ by more than tol.
    A return value of None means *identical within tolerance*.
    """
    diff = compare_images(str(baseline), str(result), tol=tol)
    assert diff is None, f"Images differ: {diff}"

def test_plot_gmm_decision_regions_visual_and_logic(
    tmp_path: Path,
    baseline_dir: Path,
    synthetic_df: pd.DataFrame,
):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    gmm, labels, boundary = figure_utils.plot_gmm_decision_regions(
        df=synthetic_df,
        x_col="X",
        y_col="Y",
        phase_col="Phase",
        ax=ax,
        xrange=[0, 20],
        yrange=[0, 20],
        decision_alpha=1.0,
    )
    assert labels.shape == (len(synthetic_df),)
    assert boundary is None or boundary.shape[1] == 2
    assert hasattr(gmm, "predict")

    out_png = tmp_path / "gmm_decision_regions.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    assert_same_image(
        result=out_png,
        baseline=baseline_dir / "gmm_decision_regions.png",
    )


def test_plot_gmm_composition_phase_visual_and_logic(
    tmp_path: Path,
    baseline_dir: Path,
    synthetic_df: pd.DataFrame,
):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    figure_utils.plot_gmm_composition_phase(
        df=synthetic_df,
        x_col="X",
        y_col="Y",
        phase_col="Phase",
        ax=ax,
        point_cmap=["#FFFFCC", "dodgerblue"],
    )
    scatters = [c for c in ax.collections if np.asarray(c.get_offsets()).size > 0]
    assert len(scatters) == 2

    out_png = tmp_path / "gmm_composition_phase.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    assert_same_image(
        result=out_png,
        baseline=baseline_dir / "gmm_composition_phase.png",
    )

@pytest.mark.parametrize(
    "writer, fname, extra_kwargs",
    [
        (
            pmf.titration_diagram,
            "titration_diagram.png",
            dict(x_col="X", y_col="Y", phase_col="Phase", xrange=[0, 20], yrange=[0, 20]),
        ),
        (
            pmf.phase_diagram_exp,
            "phase_diagram_exp.png",
            dict(x_col="X", y_col="Y", phase_col="Phase", xrange=[0, 20], yrange=[0, 20]),
        ),
        (
            pmf.mathematical_model,
            "mathematical_model.png",
            dict(x_col="X", y_col="Y", phase_col="Phase", xrange=[0, 20], yrange=[0, 20]),
        ),
    ],
)
def test_visual_regression_on_helpers(
    tmp_path: Path,
    baseline_dir: Path,
    synthetic_df: pd.DataFrame,
    writer: Callable,
    fname: str,
    extra_kwargs: dict,
):
    
    csv = tmp_path / "input.csv"
    synthetic_df.to_csv(csv, index=False)

    if writer.__name__ == "mathematical_model":
        json_file_path = tmp_path / "test_params.json"
        test_params = {
          "MODEL_A": 0.955,
          "MODEL_B": -5.73,
          "MODEL_C": 581,
          "MODEL_R2": 0.9993
        }
        with open(json_file_path, 'w') as f:
            json.dump(test_params, f)
        extra_kwargs["json_path"] = str(json_file_path)

    out_png = tmp_path / fname
    writer(file_path=str(csv), output_path=str(out_png), **extra_kwargs)

    assert out_png.is_file()
    arr = plt.imread(out_png)
    assert np.ptp(arr[..., :3]) > 0.0

    assert_same_image(out_png, baseline_dir / fname)


def test_plot_two_scatter_visual_regression(tmp_path: Path, baseline_dir: Path):
    xlsx = tmp_path / "scatter.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        pd.DataFrame({"X": [1, 2, 3], "Y": [3, 2, 1]}).to_excel(
            w, sheet_name="A", index=False
        )
        pd.DataFrame({"X": [1, 2, 3], "Y": [1, 2, 3]}).to_excel(
            w, sheet_name="B", index=False
        )

    out_png = tmp_path / "plot_two_scatter.png"
    pmf.plot_two_scatter(
        excel_file=str(xlsx),
        sheet1="A",
        sheet2="B",
        output_path=str(out_png),
        xlim=[0, 4],
        ylim=[0, 4],
    )

    assert_same_image(out_png, baseline_dir / "plot_two_scatter.png")
