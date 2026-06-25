from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import matplotlib as mpl
mpl.use("Agg")
from matplotlib.testing.compare import compare_images

from functools import partial
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif

import neat_ml.model.feature_importance as fi


@pytest.fixture(scope="module")
def classification_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic binary-classification data."""
    X_arr, y = make_classification(n_samples=10, n_features=5, n_informative=3, random_state=0)
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
        # case where the features have clear separation in rank
        (
            np.array([[0.9, 0.7, 0.0, 0.8],
                      [0.5, 0.6, 0.9, 0.7],
                      [0.4, 0.1, 0.3, 0.6]]),
            np.asarray([f"Feat_{i}" for i in range(4)]),
            2,
            ["Feat_3", "Feat_0", "Feat_2", "Feat_1"],
            [3, 2, 1, 0],
        ),
        # case where the top features have an equal number of votes.
        # the ranking follows the order of the input feature names
        (
            np.array([[0.9, 0.7, 0.0, 0.8],
                      [0.5, 0.6, 0.9, 0.7],
                      [0.4, 0.1, 0.6, 0.3]]),
            np.asarray([f"Feat_{i}" for i in range(4)]),
            2,
            ["Feat_0", "Feat_2", "Feat_3", "Feat_1"],
            [2, 2, 2, 0],
        ),
    ],
)
def test_feature_importance_consensus(
    pos_vals: np.ndarray,
    feat_names: np.ndarray,
    top_k: int,
    exp_names: list[str],
    exp_counts: list[int],
) -> None:
    """Validate the consensus ranking: names, counts, and number of models."""
    ranked_names, ranked_counts, n_models = fi.feature_importance_consensus(pos_vals, feat_names, top_k)
    assert n_models == len(pos_vals)
    assert_array_equal(ranked_names, exp_names)
    assert_array_equal(ranked_counts, exp_counts)


def test_get_k_best_scores_values_and_shapes(
    classification_dataset: tuple[pd.DataFrame, pd.Series],
) -> None:
    """
    Validate that get_k_best_scores returns one score vector per metric with:
    shape (n_features,), no NaNs
    values equal to the underlying metric outputs.
    """
    X, y = classification_dataset
    metrics = [f_classif, partial(mutual_info_classif, random_state=0)]
    out = fi.get_k_best_scores(X, y, k=3, metrics=metrics)

    assert isinstance(out, list)
    assert len(out) == len(metrics)

    for arr in out:
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (X.shape[1],)
        assert np.all(np.isfinite(arr))

    f_scores, _ = f_classif(X.to_numpy(), y)
    assert_allclose(out[0], f_scores)

    mi_scores = mutual_info_classif(X.to_numpy(), y, random_state=0)
    assert_allclose(out[1], mi_scores)


def test_plot_feat_import_consensus_image(tmp_path: Path, stable_rc, baseline_dir):
    """Image regression for the consensus plot."""
    ranked_names = np.asarray(
        ["PEO 10 kg/mol (wt%)", "Dextran 10 kg/mol (wt%)", "num_blobs", "coverage_percentage"]
    )
    ranked_counts = np.asarray([4, 3, 2, 1])

    with mpl.rc_context(stable_rc):
        fi.plot_feat_import_consensus(
            ranked_names,
            ranked_counts,
            num_models=4,
            top_feat_count=3,
            out_dir=tmp_path
        )

    actual = tmp_path/"feat_imp_consensus.png"

    expected = baseline_dir / "feat_imp_consensus_expected.png"
    result = compare_images(expected, actual, tol=1e-4) # type: ignore[call-overload]
    assert result is None

def test_compare_methods_end_to_end(
    tmp_path: Path,
    classification_dataset: tuple[pd.DataFrame, pd.Series],
    stable_rc,
    baseline_dir,
):
    """
    End-to-end test of compare_methods.
    Test consistency of mean rank of important features
    PNG compared via inline NumPy RMS diff.
    """
    rng = np.random.default_rng(0)
    X, y = classification_dataset
    # "preprocess" dataset to remove composition columns
    X = X.drop(columns=["PEO 10 kg/mol (wt%)", "Dextran 10 kg/mol (wt%)"])
    model = RandomForestClassifier(random_state=0).fit(X, y)

    with mpl.rc_context(stable_rc):
        fi.compare_methods(model, X, y, out_dir=tmp_path, top=3, rng=rng)

    actual_csv_path = tmp_path / "feature_importance_comparison.csv"

    actual_df = pd.read_csv(actual_csv_path, index_col=0)
    # SHAP importance values fluctuate on the order of 1e-2 floating
    # point precision between calls, so check that the mean ranking of
    # the feature importance values is preserved.
    assert_allclose(actual_df["mean_rank"], [1.0, 2.333333333333333, 2.6666666666666665])

    # check the output of ebm importance ranking.
    # for the same reason that SHAP values are difficult to compare,
    # the SHAP plot and FIC plots also fluctuate between runs, 
    # by a floating point value big enough to make image comparison difficult.
    ebm_act = tmp_path / "ebm_importance.png"
    ebm_exp = baseline_dir / "ebm_importance_expected.png"
    result = compare_images(ebm_exp, ebm_act, tol=1e-4) # type: ignore[call-overload]
    assert result is None


def test_plot_feature_importance_comparsion(tmp_path, baseline_dir):
    """
    regression test for the visual appearance of
    ``feature_importance_comparison.png``
    """
    comp_df = pd.read_csv(baseline_dir / "feature_importance_comparison_expected.csv")
    fi.plot_feature_importance_comparison(comp_df, tmp_path, top=3)
    output_img = tmp_path / "feature_importance_comparison.png"
    exp_img = baseline_dir / "feature_importance_comparison_expected.png"
    result = compare_images(exp_img, output_img, tol=1e-4)
    assert result is None
