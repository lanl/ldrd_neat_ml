from pathlib import Path
import os
import random
import shutil
from typing import List, Tuple

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from matplotlib.testing.compare import compare_images
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

import neat_ml.model.feature_importance as fi

@pytest.fixture(scope="module")
def classification_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """Synthetic binary-classification data."""
    X_arr, y = make_classification(
        n_samples=200, n_features=5, n_informative=3, random_state=0
    )
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


@pytest.fixture(scope="session")
def baseline_dir() -> Path:
    """Directory that stores baseline artefacts (PNGs + CSV)."""
    return Path(__file__).parent / "baseline"

def assert_images_equal(
    desired: Path, 
    actual: Path, 
    tol: float = 1e-4
):
    """Fail if *actual* and *desired* images differ beyond tol RMS."""
    if not desired.is_file():
        shutil.copyfile(actual, desired)
        pytest.skip(f"Baseline image {desired.name} created - re-run tests")

    diff = compare_images(str(desired), str(actual), tol=tol)
    err_msg = "" if diff is None else f"Image mismatch: {diff}"
    npt.assert_equal(diff, None, err_msg=err_msg)

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
    ranked_names, ranked_counts, n_models = fi.feature_importance_consensus(
        pos_vals, feat_names, top_k
    )
    npt.assert_equal(n_models, len(pos_vals))
    npt.assert_array_equal(ranked_names, exp_names)
    npt.assert_array_equal(ranked_counts, exp_counts)


def test_plot_feat_import_consensus_image(
    tmp_path: Path, 
    baseline_dir: Path
):
    """Image diff test kept for the standalone consensus plot."""
    ranked_names = np.asarray(
        [
            "PEO 10 kg/mol (wt%)",
            "Dextran 10 kg/mol (wt%)",
            "num_blobs",
            "coverage_percentage",
        ]
    )
    ranked_counts = np.asarray([4, 3, 2, 1])

    fi.plot_feat_import_consensus(
        ranked_names, ranked_counts, num_models=4, top_feat_count=3, out_dir=tmp_path
    )

    actual = tmp_path / "feat_imp_consensus.png"
    desired = baseline_dir / "feat_imp_consensus_expected.png"
    assert_images_equal(desired, actual, tol=1e-4)


def test_compare_methods_end_to_end(
    tmp_path: Path, 
    classification_dataset, 
    baseline_dir: Path
):
    """
    End-to-end test of compare_methods.
    Only the CSV is compared; images are just checked for existence.
    """

    if any(
        lib is None
        for lib in (
            fi.shap,
            fi.ExplainableBoostingClassifier,
            fi.LimeTabularExplainer,
        )
    ):
        pytest.skip("Optional SHAP / interpret / LIME libraries not installed")

    random.seed(0)
    np.random.seed(0)
    os.environ["PYTHONHASHSEED"] = "0"

    X, y = classification_dataset
    model = RandomForestClassifier(random_state=0).fit(X, y)
    fi.compare_methods(model, X, y, out_dir=tmp_path, top=3)

    actual_csv_path   = tmp_path / "feature_importance_comparison.csv"
    baseline_csv_path = baseline_dir / "feature_importance_comparison_expected.csv"

    expected_file_exists = True
    actual_file_exists   = actual_csv_path.is_file()
    npt.assert_equal(actual_file_exists, expected_file_exists)

    actual_df   = (
        pd.read_csv(actual_csv_path, index_col=0)
        .sort_index()
        .sort_index(axis=1)
    )
    if not baseline_csv_path.is_file():
        actual_df.to_csv(baseline_csv_path)
        pytest.skip("Baseline CSV created - re-run the suite to enable comparison")

    expected_df = (
        pd.read_csv(baseline_csv_path, index_col=0)
        .sort_index()
        .sort_index(axis=1)
    )

    pd.testing.assert_frame_equal(
        actual_df,
        expected_df,
        check_like=True,     # ignore column order (already sorted, but explicit)
        check_exact=False,   # allow floating‑point tolerance below
        rtol=2e-2,           # 2 % relative tolerance accommodates LIME noise
        atol=0,
    )

    for png_name in (
        "feature_importance_comparison.png",
    ):  
        actual_png = tmp_path / png_name
        baseline_png = baseline_dir / f"{png_name[:-4]}_expected.png"
        assert_images_equal(baseline_png, actual_png, tol=1e-4)
