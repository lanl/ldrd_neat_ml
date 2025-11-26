from pathlib import Path
from importlib import resources
from typing import Callable, Generator, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from matplotlib.testing.compare import compare_images
from sklearn.mixture import GaussianMixture

from neat_ml.utils import figure_utils

@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """Deterministic (seeded) composition/phase dataframe."""
    rng = np.random.default_rng(7)
    x = rng.uniform(0.0, 20.0, 30)
    y = rng.uniform(0.0, 20.0, 30)
    phase = (x + y > 20.0).astype(int)
    return pd.DataFrame({"Sodium Citrate (wt%)": x,
                         "PEO 8 kg/mol (wt%)": y,
                         "Phase": phase})

@pytest.fixture(scope="session")
def baseline_dir() -> Generator[Any, Any, Any]:
    """
    Directory that stores the reference (expected) images.
    """
    ref = resources.files("neat_ml.tests") / "baseline"
    with resources.as_file(ref) as path:
        yield path

def test_standardise_labels_mapping():
    raw = np.array([9, 9, 1])
    x_comp = np.array([[10, 10], [11, 11], [0.1, 0.1]])
    actual_std, mapping = figure_utils._standardise_labels(raw, x_comp)
    
    assert mapping[1] == 1
    assert mapping[9] == 0
    
    desired_std = np.array([0, 0, 1])
    npt.assert_array_equal(actual_std, desired_std)

@pytest.mark.parametrize("z, exp_out",
    [
        (
            np.array([[0, 1],
                      [1, 0]]),
            np.array([[1., 0.5],
                      [0.5, 0.0]]),
        ),
        (
            np.array([[0, 0],
                      [0, 0]]),
            None,
        ),
    ]
)
def test_extract_boundary_from_contour(z, exp_out):
    xs = ys = np.arange(2)
    boundary = figure_utils.extract_boundary_from_contour(z, xs, ys, level=0.5)
    
    if exp_out is None:
        assert boundary is exp_out
    else:
        npt.assert_array_equal(boundary, exp_out)


def test_gmmwrapper_predict_matches_gmm():
    rng = np.random.default_rng(0)
    feats = rng.uniform(0, 1, (20, 1))
    gmm = GaussianMixture(n_components=2, random_state=0).fit(feats)
    x_comp = rng.uniform(0, 20, (20, 2))

    wrapper = figure_utils.GMMWrapper(gmm, x_comp, feats)
    desired = gmm.predict(feats)
    actual = wrapper.predict(x_comp)

    npt.assert_array_equal(actual, desired)


def test_set_axis_style_equal_aspect():
    fig, ax = plt.subplots()
    figure_utils._set_axis_style(ax, [0, 10], [0, 5])
    assert ax.get_aspect() == 'auto'
    plt.close(fig)

@pytest.mark.parametrize(
    "plotter, fname, region_colors",
    [
        (
            figure_utils.plot_gmm_decision_regions,
            "gmm_decision_regions.png",
            ["lightsteelblue", "aquamarine"],
        ),
        (
            figure_utils.plot_gmm_decision_regions,
            "gmm_decision_regions.png",
            None,
        ),
        (
            figure_utils.plot_gmm_composition_phase,
            "gmm_composition_phase.png",
            ["#FF8C00", "dodgerblue"],
        ),
        (
            figure_utils.plot_gmm_composition_phase,
            "gmm_composition_phase.png",
            None,
        ),
    ],
)
def test_plotters_visual_and_logic(
    tmp_path: Path,
    synthetic_df: pd.DataFrame,
    baseline_dir: Path,
    plotter: Callable,
    fname: str,
    region_colors: Optional[list[str]],
):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

    if plotter is figure_utils.plot_gmm_decision_regions:
        gmm, labels, boundary = plotter(
            df=synthetic_df,
            x_col=synthetic_df.columns[0],
            y_col=synthetic_df.columns[1],
            phase_col="Phase",
            ax=ax,
            xrange=[0, 20],
            yrange=[0, 20],
            n_components=2,
            random_state=42,
            boundary_color="red",
            resolution=200,
            decision_alpha=1,
            plot_regions=True,
            region_colors=region_colors,
        )
        assert labels.shape == (len(synthetic_df),)
        assert boundary is None or boundary.shape[1] == 2
        assert hasattr(gmm, "predict")

    else: 
        plotter(
            df=synthetic_df,
            x_col=synthetic_df.columns[0],
            y_col=synthetic_df.columns[1],
            phase_col="Phase",
            ax=ax,
            point_cmap=region_colors,
        )
        scat = [c for c in ax.collections if np.asarray(c.get_offsets()).size > 0]
        assert len(scat) == 2

    out_png = tmp_path / fname
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    # here ``compare_images`` ``tol`` was loosened from 1e-4 to
    # 2e-2 to accommodate cross-platform testing as discussed at
    # https://github.com/lanl/ldrd_neat_ml/pull/1#issuecomment-3463967764
    result = compare_images(str(baseline_dir / fname), str(out_png), tol=2e-2)
    assert result is None


@pytest.mark.parametrize("column_name, out_column_name",
    [
        ("Dextran 9 - 11 kg/mol (wt%)", "Dextran 10 kg/mol (wt%)"),
        ("Dextran 450 - 650 kg/mol (wt%)", "Dextran 500 kg/mol (wt%)"),
        ("Sodium citrate (wt%)", "Sodium Citrate (wt%)"),
        ("PEO 8 kg/mol (wt%)", "PEG 8 kg/mol (wt%)"),
    ]
)
def test_rename_df_columns(column_name, out_column_name):
    df_in = pd.DataFrame(columns=[column_name])
    df_out, x_col = figure_utils.rename_df_columns(df_in, column_name)
    assert df_out.columns == x_col == out_column_name
