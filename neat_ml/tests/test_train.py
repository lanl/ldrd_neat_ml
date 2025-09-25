import os
from pathlib import Path
from importlib import resources

import joblib
import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
from matplotlib.testing.compare import compare_images

from numpy import testing as npt
from pandas.api.types import is_integer_dtype
from sklearn.pipeline import Pipeline

from neat_ml.model.train import (
    _build_pipeline,
    _scale_pos_weight,
    plot_roc,
    preprocess,
    save_model_bundle,
    train_with_validation,
)

@pytest.fixture
def sample_data() -> pd.DataFrame:
    """
    Provides a sample DataFrame for consistent testing.
    """
    rng = np.random.default_rng(42)
    data = {
        "feature1": rng.random(100),
        "feature2": rng.random(100) * 10,
        "feature3": ["A"] * 50 + ["B"] * 50,
        "exclude_col": np.arange(100),
        "target": rng.integers(0, 2, 100),
    }
    df = pd.DataFrame(data)
    df.loc[5, "feature1"] = np.nan
    df.loc[10, "target"] = np.nan
    return df


def test_preprocess(sample_data: pd.DataFrame):
    actual_X, actual_y = preprocess(
        sample_data,
        target="target",
        exclude=["exclude_col"],
    )

    assert isinstance(actual_X, pd.DataFrame)
    assert isinstance(actual_y, pd.Series)
    npt.assert_equal(actual_X.shape[0], 99)
    npt.assert_equal(actual_y.shape[0], 99)
    npt.assert_("target" not in actual_X.columns)
    npt.assert_("exclude_col" not in actual_X.columns)
    npt.assert_(not actual_X.isnull().any().any())
    npt.assert_(not actual_y.isnull().any())
    npt.assert_(is_integer_dtype(actual_y))

def test_preprocess_no_exclude(sample_data: pd.DataFrame):
    actual_X, _ = preprocess(sample_data, target="target")
    npt.assert_("exclude_col" in actual_X.columns)
    npt.assert_("target" not in actual_X.columns)


def test_scale_pos_weight():
    y_imbalanced = pd.Series([0, 0, 0, 0, 1, 1])
    npt.assert_allclose(_scale_pos_weight(y_imbalanced), 2.0)

    y_balanced = pd.Series([0, 1, 0, 1])
    npt.assert_allclose(_scale_pos_weight(y_balanced), 1.0)

    y_no_pos = pd.Series([0, 0, 0, 0])
    npt.assert_allclose(_scale_pos_weight(y_no_pos), 1.0)


def test_build_pipeline():
    actual_pipeline = _build_pipeline(scale_pos_weight=1.5)
    expected_steps = ["impute", "scale", "ensemble"]

    assert isinstance(actual_pipeline, Pipeline)
    npt.assert_equal(list(actual_pipeline.named_steps.keys()), expected_steps)


def test_train_with_validation(sample_data: pd.DataFrame):
    X, y = preprocess(sample_data, target="target")
    X_train, y_train = X.iloc[:80], y.iloc[:80]
    X_val, y_val = X.iloc[80:], y.iloc[80:]

    model, metrics, _, actual_val_proba = train_with_validation(
        X_train, y_train, X_val, y_val
    )

    assert isinstance(model, Pipeline)
    assert isinstance(metrics, dict)
    assert isinstance(actual_val_proba, np.ndarray)
    npt.assert_("val_roc_auc" in metrics)
    npt.assert_("val_pr_auc" in metrics)
    npt.assert_(0.0 <= metrics["val_roc_auc"] <= 1.0)
    npt.assert_equal(actual_val_proba.shape[0], X_val.shape[0])


def test_plot_roc(tmp_path: Path):
    
    pkg_root = resources.files(__package__)
    baseline_res = pkg_root/"baseline"
    
    with resources.as_file(baseline_res) as baseline_path:
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.6])
        expected_image_path = baseline_path/"expected_train_roc.png"
        actual_image_path = tmp_path/"actual_train_roc.png"
        plot_roc(y_true, y_prob, out_png=str(actual_image_path), label="Test ROC")
        compare_images(str(expected_image_path), str(actual_image_path), tol=1e-4)


def test_save_model_bundle(tmp_path: Path, sample_data: pd.DataFrame):
    X, y = preprocess(sample_data, target="target")
    X_train, y_train = X.iloc[:80], y.iloc[:80]
    X_val, y_val = X.iloc[80:], y.iloc[80:]
    expected_model, expected_metrics, expected_params, _ = train_with_validation(
        X_train, y_train, X_val, y_val
    )

    features = list(X.columns)
    bundle_path = tmp_path / "model.joblib"

    save_model_bundle(
        expected_model,
        features,
        expected_metrics,
        expected_params,
        str(bundle_path),
    )

    npt.assert_(os.path.exists(bundle_path))
    actual_bundle = joblib.load(bundle_path)

    assert isinstance(actual_bundle, dict)
    npt.assert_equal(
        sorted(list(actual_bundle.keys())),
        sorted(["model", "features", "metrics", "best_params"]),
    )
    assert isinstance(actual_bundle["model"], Pipeline)
    npt.assert_equal(actual_bundle["features"], features)
    npt.assert_equal(actual_bundle["metrics"], expected_metrics)
    npt.assert_equal(actual_bundle["best_params"], expected_params)
