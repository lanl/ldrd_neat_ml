from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.testing.compare import compare_images
from sklearn.mixture import GaussianMixture

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


def assert_same_image(result: Path, baseline: Path, *, tol: float = 10.0) -> None:
    diff = compare_images(str(baseline), str(result), tol=tol)
    assert diff is None, f"Images differ: {diff}"

pytestmark = pytest.mark.usefixtures("baseline_dir", "synthetic_df")

def test_safe_read_excel_raises(tmp_path: Path) -> None:
    non_existent = tmp_path / "missing.xlsx"
    with pytest.raises(FileNotFoundError):
        figure_utils._safe_read_excel(non_existent, sheet_name="foo")

def test_axis_ranges() -> None:
    df_a = pd.DataFrame({"x": [1, 3], "y": [2, 5]})
    df_b = pd.DataFrame({"x": [0, 7], "y": [1, 4]})
    xr, yr = figure_utils._axis_ranges(df_a, df_b, "x", "y", pad=1)
    assert xr == [0, 7 + 1]
    assert yr == [0, 5 + 1]


def test_standardise_labels_mapping() -> None:
    raw = np.array([9, 9, 1])
    x_comp = np.array([[10, 10], [11, 11], [0.1, 0.1]])
    std, mapping = figure_utils._standardise_labels(raw, x_comp)
    assert mapping[1] == 1
    assert mapping[9] == 0
    assert std.tolist() == [0, 0, 1]


def test_extract_boundary_from_contour() -> None:
    z = np.array([[0, 1],
                  [1, 0]])
    xs = ys = np.arange(2)
    boundary = figure_utils.extract_boundary_from_contour(z, xs, ys, level=0.5)
    assert boundary is not None and boundary.shape[1] == 2


def test_gmmwrapper_predict_matches_gmm() -> None:
    rng = np.random.default_rng(0)
    feats = rng.uniform(0, 1, (20, 1))
    gmm = GaussianMixture(n_components=2, random_state=0).fit(feats)
    x_comp = rng.uniform(0, 20, (20, 2))

    wrapper = figure_utils.GMMWrapper(gmm, x_comp, feats)
    preds_direct = gmm.predict(feats)
    preds_wrapper = wrapper.predict(x_comp)

    assert np.array_equal(preds_direct, preds_wrapper)


def test_set_axis_style_equal_aspect() -> None:
    fig, ax = plt.subplots()
    figure_utils._set_axis_style(ax, [0, 10], [0, 5])
    assert ax.get_aspect() == 1.0
    plt.close(fig)

@pytest.mark.parametrize(
    "plotter, fname",
    [
        (figure_utils.plot_gmm_decision_regions, "gmm_decision_regions.png"),
        (figure_utils.plot_gmm_composition_phase, "gmm_composition_phase.png"),
    ],
)
def test_plotters_visual_and_logic(
    tmp_path: Path,
    baseline_dir: Path,
    synthetic_df: pd.DataFrame,
    plotter: Callable,
    fname: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

    if plotter is figure_utils.plot_gmm_decision_regions:
        gmm, labels, boundary = plotter(
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

    else: 
        plotter(
            df=synthetic_df,
            x_col="X",
            y_col="Y",
            phase_col="Phase",
            ax=ax,
            point_cmap=["#FFFFCC", "dodgerblue"],
        )
        scat = [c for c in ax.collections if np.asarray(c.get_offsets()).size > 0]
        assert len(scat) == 2

    out_png = tmp_path / fname
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    assert_same_image(out_png, baseline_dir / fname)
