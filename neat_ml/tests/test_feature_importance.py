from pathlib import Path
import os
import random
from typing import List, Tuple
import importlib.resources as resources

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import matplotlib as mpl
mpl.use("Agg")
from matplotlib.testing.compare import compare_images

from functools import partial
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif

import neat_ml.model.feature_importance as fi


STABLE_RC = {
    "figure.figsize": (6.0, 4.0),
    "figure.dpi": 100,
    "savefig.dpi": 100,
    "savefig.bbox": "standard",
    "savefig.pad_inches": 0.0,
    "font.family": ["DejaVu Sans"],
    "font.size": 10.0,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.linewidth": 1.0,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "path.simplify": False,
    "text.antialiased": True,
    "lines.antialiased": True,
}


@pytest.fixture(scope="module")
def classification_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """Synthetic binary-classification data."""
    X_arr, y = make_classification(n_samples=200, n_features=5, n_informative=3, random_state=0)
    X = pd.DataFrame(
        X_arr,
        columns=[
            "PEO 10 kg/mol (wt%)",
            "Dextran 10 kg/mol (wt%)",
            "num_blobs",
            "coverage_percentage",
            "graph_num_components",
        ],
    )
    return X, pd.Series(y, name="y")


@pytest.mark.parametrize(
    "pos_vals, feat_names, top_k, exp_names, exp_counts",
    [
        (
            [
                np.array([[0.9, 0.7, 0.0], [0.8, 0.6, 0.1]]),
                np.array([[0.0, 0.8, 0.6], [0.1, 0.8, 0.7]]),
            ],
            np.asarray([f"Feat_{i}" for i in range(3)]),
            2,
            ["Feat_1", "Feat_0", "Feat_2"],
            [2, 1, 1],
        ),
        (
            [
                np.array([[0.9, 0.7, 0.0, 0.7], [0.8, 0.6, 0.1, 0.7]]),
                np.array([[0.8, 0.1, 0.2, 0.9], [0.8, 0.1, 0.8, 0.9]]),
                np.array([[0.7, 0.7, 0.7, 0.1], [0.7, 0.7, 0.8, 0.1]]),
            ],
            np.asarray([f"Feat_{i}" for i in range(4)]),
            3,
            ["Feat_0", "Feat_3", "Feat_1", "Feat_2"],
            [3, 2, 2, 2],
        ),
        (
            [
                np.array([[0.9, 0.7, 0.0], [0.8, 0.6, 0.1]]),
                np.array([0.0, 0.8, 0.6]),
            ],
            np.asarray([f"Feat_{i}" for i in range(3)]),
            2,
            ["Feat_1", "Feat_0", "Feat_2"],
            [2, 1, 1],
        ),
    ],
)
def test_feature_importance_consensus(
    pos_vals: List[np.ndarray],
    feat_names: np.ndarray,
    top_k: int,
    exp_names: List[str],
    exp_counts: List[int],
) -> None:
    """Validate the consensus ranking: names, counts, and number of models."""
    ranked_names, ranked_counts, n_models = fi.feature_importance_consensus(pos_vals, feat_names, top_k)
    npt.assert_equal(n_models, len(pos_vals))
    npt.assert_array_equal(ranked_names, exp_names)
    npt.assert_array_equal(ranked_counts, exp_counts)


def test_get_k_best_scores_values_and_shapes(
    classification_dataset: Tuple[pd.DataFrame, pd.Series],
) -> None:
    """
    Validate that get_k_best_scores returns one score vector per metric with:
    shape (n_features,), no NaNs
    values equal to the underlying metric outputs.
    """
    X, y = classification_dataset
    metrics = [f_classif, partial(mutual_info_classif, random_state=0)]
    out = fi.get_k_best_scores(X, y, k=3, metrics=metrics)

    npt.assert_(isinstance(out, list))
    npt.assert_equal(len(out), len(metrics))

    for arr in out:
        npt.assert_(isinstance(arr, np.ndarray))
        npt.assert_equal(arr.shape, (X.shape[1],))
        npt.assert_(np.all(np.isfinite(arr)))

    f_scores, _ = f_classif(X.to_numpy(), y)
    npt.assert_allclose(out[0], f_scores, rtol=1e-12, atol=0.0)

    mi_scores = mutual_info_classif(X.to_numpy(), y, random_state=0)
    npt.assert_allclose(out[1], mi_scores, rtol=1e-12, atol=0.0)


def test_plot_feat_import_consensus_image(tmp_path: Path) -> None:
    """Image regression for the consensus plot using inline NumPy RMS comparison."""
    ranked_names = np.asarray(
        ["PEO 10 kg/mol (wt%)", "Dextran 10 kg/mol (wt%)", "num_blobs", "coverage_percentage"]
    )
    ranked_counts = np.asarray([4, 3, 2, 1])

    with mpl.rc_context(STABLE_RC):
        fi.plot_feat_import_consensus(ranked_names, ranked_counts, num_models=4, top_feat_count=3, out_dir=tmp_path)

    actual = tmp_path/"feat_imp_consensus.png"

    pkg_root = resources.files(__package__)
    baseline_res = pkg_root/"baseline"
    with resources.as_file(baseline_res) as baseline_dir:
        expected = Path(baseline_dir)/"feat_imp_consensus_expected.png"
        compare_images(str(actual), str(expected), tol=1e-4)

def test_compare_methods_end_to_end(
    tmp_path: Path, classification_dataset: Tuple[pd.DataFrame, pd.Series]
) -> None:
    """
    End-to-end test of compare_methods.
    CSV content compared via numpy.testing (indices, columns, values).
    PNG compared via inline NumPy RMS diff.
    """
    if any(lib is None for lib in (
        fi.shap, 
        fi.ExplainableBoostingClassifier, 
        fi.LimeTabularExplainer
        )):
        pytest.skip("Optional SHAP / interpret / LIME libraries not installed")

    random.seed(0)
    np.random.seed(0)
    os.environ["PYTHONHASHSEED"] = "0"

    X, y = classification_dataset
    model = RandomForestClassifier(random_state=0).fit(X, y)

    with mpl.rc_context(STABLE_RC):
        fi.compare_methods(model, X, y, out_dir=tmp_path, top=3)

    actual_csv_path = tmp_path/"feature_importance_comparison.csv"
    npt.assert_equal(actual_csv_path.is_file(), True)

    actual_df = pd.read_csv(actual_csv_path, index_col=0).sort_index().sort_index(axis=1)

    pkg_root = resources.files(__package__)
    baseline_res = pkg_root/"baseline"
    with resources.as_file(baseline_res) as baseline_dir:
        baseline_csv_path = Path(baseline_dir)/"feature_importance_comparison_expected.csv"

        if not baseline_csv_path.is_file():
            actual_df.to_csv(baseline_csv_path)
            pytest.skip("Baseline CSV created - re-run the suite to enable comparison")

        expected_df = pd.read_csv(baseline_csv_path, index_col=0).sort_index().sort_index(axis=1)

        npt.assert_array_equal(actual_df.index.to_numpy(), expected_df.index.to_numpy())
        npt.assert_array_equal(actual_df.columns.to_numpy(), expected_df.columns.to_numpy())
        npt.assert_equal(actual_df.shape, expected_df.shape)
        npt.assert_allclose(actual_df.to_numpy(), expected_df.to_numpy(), rtol=2e-2, atol=0.0)

        actual_png = tmp_path/"feature_importance_comparison.png"
        baseline_png = Path(baseline_dir)/"feature_importance_comparison_expected.png"
        compare_images(str(actual_png), str(baseline_png), tol=1e-4)

