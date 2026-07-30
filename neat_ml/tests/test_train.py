from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_equal

import matplotlib
matplotlib.use("Agg")
from matplotlib.testing.compare import compare_images

from sklearn.pipeline import Pipeline

from neat_ml.model.train import (
    _scale_pos_weight,
    plot_roc,
    preprocess,
    save_model_bundle,
    train_with_validation,
)


def test_preprocess(sample_data: pd.DataFrame):
    actual_X, actual_y = preprocess(
        sample_data,
        target="target",
        exclude=["exclude_col"],
    )

    assert isinstance(actual_X, pd.DataFrame)
    assert isinstance(actual_y, pd.Series)
    assert actual_X.shape[0] == actual_y.shape[0] == 99 
    assert "target" not in actual_X.columns
    assert "exclude_col" not in actual_X.columns
    assert actual_X.isnull().sum().sum() == 0
    assert not actual_y.isnull().any()
    assert actual_y.dtype == int

def test_preprocess_no_exclude(sample_data: pd.DataFrame):
    actual_X, _ = preprocess(sample_data, target="target")
    assert "exclude_col" in actual_X.columns
    assert "target" not in actual_X.columns


@pytest.mark.parametrize("y_in, exp",
    [
        ([0, 0, 0, 0, 1, 1], 2.0),
        ([0, 1, 0, 1], 1.0),
        ([0, 0, 0, 0], 1.0),
    ]
)
def test_scale_pos_weight(y_in, exp):
    y_out = pd.Series(y_in)
    assert_allclose(_scale_pos_weight(y_out), exp)


def test_train_with_validation(sample_data: pd.DataFrame):
    X, y = preprocess(sample_data, target="target")
    # perfectly align all the feature data with the target
    rng = np.random.default_rng(123)
    X['feature1'] = np.where(
        y == 1.0,
        rng.uniform(0.6, 1.0, len(X)),
        rng.uniform(0.0, 0.4, len(X))
    )
    X['feature2'] = np.where(
        y == 1.0,
        rng.uniform(6, 10, len(X)),
        rng.uniform(0, 4, len(X))
    )
    X_train, y_train = X.iloc[:80], y.iloc[:80]
    X_val, y_val = X.iloc[80:], y.iloc[80:]

    model, metrics, _, actual_val_proba = train_with_validation(
        X_train, y_train, X_val, y_val, n_jobs=1, ml_hyper_opt=False,
    )

    assert isinstance(model, Pipeline)
    assert isinstance(metrics, dict)
    assert isinstance(actual_val_proba, np.ndarray)
    assert "val_roc_auc" in metrics
    assert "val_pr_auc" in metrics
    assert metrics["val_roc_auc"] == 1.0
    assert actual_val_proba.shape[0] == X_val.shape[0]
    assert_allclose(
        actual_val_proba[:5],
        np.array([0.52166154, 0.01328962, 0.94898656, 0.01328962, 0.01328962])
    )


def test_plot_roc(tmp_path: Path, baseline_dir):
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.6])
    expected_image_path = baseline_dir / "expected_train_roc.png"
    actual_image_path = tmp_path / "actual_train_roc.png"
    plot_roc(y_true, y_prob, out_png=str(actual_image_path), label="Test")
    result = compare_images(expected_image_path, actual_image_path, tol=1e-4) # type: ignore[call-overload]
    assert result is None

def test_save_model_bundle(tmp_path: Path, sample_data: pd.DataFrame):
    X, y = preprocess(sample_data, target="target")
    X_train, y_train = X.iloc[:80], y.iloc[:80]
    X_val, y_val = X.iloc[80:], y.iloc[80:]
    model, metrics, params, _ = train_with_validation(
        X_train, y_train, X_val, y_val, n_jobs=1, ml_hyper_opt=False
    )

    features = list(X.columns)
    bundle_path = tmp_path / "model.joblib"

    save_model_bundle(
        model,
        features,
        metrics,
        params,
        bundle_path,
    )

    actual_bundle = joblib.load(bundle_path)
    expected_metrics = {'val_roc_auc': 0.6222222222222222, 'val_pr_auc': 0.5988003663003663}

    assert isinstance(actual_bundle, dict)
    assert_equal(
        sorted(list(actual_bundle.keys())),
        sorted(["model", "features", "metrics", "best_params"]),
    )
    assert isinstance(actual_bundle["model"], Pipeline)
    assert actual_bundle["features"] == ['feature1', 'feature2', 'exclude_col']
    # compare output metrics (ROC-AUC, PR-AUC) 
    actual_metrics = actual_bundle["metrics"]
    assert actual_metrics.keys() == expected_metrics.keys()
    for actual_val, exp_val in zip(actual_metrics.values(), expected_metrics.values()):
        assert_allclose(actual_val, exp_val) 
    actual_params = actual_bundle["best_params"]
    # spot check output values against expected parameters
    assert_allclose(actual_params["ensemble__xgb__scale_pos_weight"], 0.9512195121951219)
    assert actual_params["ensemble__xgb__subsample"] == 0.8
    assert actual_params["ensemble__rf__n_estimators"] == 500
    assert actual_params["ensemble__rf__criterion"] == 'gini'
