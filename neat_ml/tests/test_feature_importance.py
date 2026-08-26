import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

import matplotlib
matplotlib.use("Agg")
from matplotlib.testing.compare import compare_images

from sklearn.ensemble import RandomForestClassifier

import neat_ml.model.feature_importance as fi


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
    pos_vals,
    feat_names,
    top_k,
    exp_names,
    exp_counts,
):
    """Validate the consensus ranking: names, counts, and number of models."""
    ranked_names, ranked_counts, n_models = fi.feature_importance_consensus(
        pos_vals, feat_names, top_k
    )
    assert n_models == len(pos_vals)
    assert_array_equal(ranked_names, exp_names)
    assert_array_equal(ranked_counts, exp_counts)


def test_plot_feat_import_consensus_image(tmp_path, stable_rc, baseline_dir):
    """Image regression for the consensus plot."""
    ranked_names = np.asarray(
        [
            "PEO 10 kg/mol (wt%)",
            "Dextran 10 kg/mol (wt%)",
            "num_blobs",
            "coverage_percentage"
        ]
    )
    ranked_counts = np.asarray([4, 3, 2, 1])

    with matplotlib.rc_context(stable_rc):
        fi.plot_feat_import_consensus(
            ranked_names,
            ranked_counts,
            num_models=4,
            top_feat_count=3,
            out_dir=tmp_path
        )

    actual = tmp_path / "feat_imp_consensus.png"

    expected = baseline_dir / "feat_imp_consensus_expected.png"
    result = compare_images(expected, actual, tol=1e-4)
    assert result is None

def test_compare_methods_end_to_end(
    tmp_path,
    classification_dataset,
    stable_rc,
    baseline_dir,
):
    """
    Test that `compare_methods` generates a CSV file that
    contains feature importance values that are ranked correctly,
    and produces reasonable output plots of feature importance.
    """
    rng = np.random.default_rng(0)
    X, y = classification_dataset
    # "preprocess" dataset to remove composition columns
    X = X.drop(columns=["PEO 10 kg/mol (wt%)", "Dextran 10 kg/mol (wt%)"])
    model = RandomForestClassifier(random_state=0).fit(X, y)

    with matplotlib.rc_context(stable_rc):
        fi.compare_methods(model, X, y, out_dir=tmp_path, top=3, rng=rng, random_seed=0)

    actual_csv_path = tmp_path / "feature_importance_comparison.csv"

    actual_df = pd.read_csv(actual_csv_path, index_col=0)
    # make assertions on the final ranking of the features
    assert_array_equal(
        actual_df.index,
        ['coverage_percentage', 'num_blobs', 'graph_num_components']
    )

    # check the output of ebm importance ranking, shap summary plot, and fic plot
    ebm_act = tmp_path / "ebm_importance.png"
    ebm_exp = baseline_dir / "ebm_importance_expected.png"
    result = compare_images(ebm_exp, ebm_act, tol=1e-4)
    shap_act = tmp_path / "shap_summary.png"
    shap_exp = baseline_dir / "shap_summary_exp.png"
    result2 = compare_images(shap_exp, shap_act, tol=1e-4)
    fic_act = tmp_path / "feat_imp_consensus.png"
    fic_exp = baseline_dir / "feat_imp_consensus_exp.png"
    result3 = compare_images(fic_exp, fic_act, tol=1e-4)
    assert result is None
    assert result2 is None
    assert result3 is None


def test_plot_feature_importance_comparsion(tmp_path, baseline_dir):
    """
    regression test for the visual appearance of
    ``feature_importance_comparison.png``
    """
    comp_df = pd.read_csv(baseline_dir / "feature_importance_comparison.csv")
    fi.plot_feature_importance_comparison(comp_df, tmp_path, top=3)
    output_img = tmp_path / "feature_importance_comparison.png"
    exp_img = baseline_dir / "feature_importance_comparison_expected.png"
    result = compare_images(exp_img, output_img, tol=1e-4)
    assert result is None
