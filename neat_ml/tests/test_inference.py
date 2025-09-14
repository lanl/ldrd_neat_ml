from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from numpy import testing as npt
from pandas import testing as pdt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from neat_ml.model.inference import run_inference, save_predictions

@pytest.fixture
def sample_inference_data(tmp_path: Path) -> Path:
    """
    Provides a sample CSV file for inference testing.
    """
    rng = np.random.default_rng(123)
    data = {
        "feat_a": rng.random(50),
        "feat_b": np.arange(50),
        "id_col": [f"id_{i}" for i in range(50)],
        "ground_truth": rng.integers(0, 2, 50),
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path/"inference_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

@pytest.fixture
def trained_model_bundle(tmp_path: Path) -> Path:
    """Creates and saves a dummy trained model bundle."""
    features = ["feat_a", "feat_b"]
    model = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(random_state=42)),
        ]
    )
    rng = np.random.default_rng(7)
    dummy_X = pd.DataFrame(rng.random((10, len(features))), columns=features)
    dummy_y = pd.Series(rng.integers(0, 2, 10))
    model.fit(dummy_X, dummy_y)

    bundle = {"model": model, "features": features}
    model_path = tmp_path/"model.joblib"
    joblib.dump(bundle, model_path)
    return model_path


def test_save_predictions(tmp_path: Path):
    expected_df_subset = pd.DataFrame({"id": [1, 2, 3], "data": ["a", "b", "c"]})
    expected_prob = np.array([0.1, 0.6, 0.9])
    expected_labels = np.array([0, 1, 1])
    actual_out_csv = tmp_path/"preds.csv"

    save_predictions(expected_df_subset, expected_prob, actual_out_csv)

    npt.assert_(actual_out_csv.exists())
    actual_df = pd.read_csv(actual_out_csv)
    actual_df_subset = actual_df[["id", "data"]]
    actual_prob = actual_df["Pred_Prob"].to_numpy()
    actual_labels = actual_df["Pred_Label"].to_numpy()

    pdt.assert_frame_equal(expected_df_subset, actual_df_subset)
    npt.assert_allclose(expected_prob, actual_prob)
    npt.assert_array_equal(expected_labels, actual_labels)

def test_run_inference_with_target(
    trained_model_bundle: Path, sample_inference_data: Path, tmp_path: Path
):
    actual_pred_csv_path = tmp_path/"predictions.csv"

    run_inference(
        model_in=trained_model_bundle,
        data_csv=sample_inference_data,
        target="ground_truth",
        exclude_cols=["id_col"],
        pred_csv=actual_pred_csv_path,
    )
    npt.assert_(actual_pred_csv_path.exists(), "Prediction CSV was not created.")

    actual_preds_df = pd.read_csv(actual_pred_csv_path)
    expected_rows = 50
    npt.assert_equal(len(actual_preds_df), expected_rows, err_msg="Output CSV has incorrect row count.")
    npt.assert_("Pred_Prob" in actual_preds_df.columns)
    npt.assert_("Pred_Label" in actual_preds_df.columns)

def test_run_inference_no_target(
    trained_model_bundle: Path, sample_inference_data: Path, tmp_path: Path
):
    actual_pred_csv_path = tmp_path/"predictions.csv"

    run_inference(
        model_in=trained_model_bundle,
        data_csv=sample_inference_data,
        target=None,
        exclude_cols=["id_col"],
        pred_csv=actual_pred_csv_path,
    )

    npt.assert_(actual_pred_csv_path.exists())


def test_run_inference_handles_missing_feature(
    trained_model_bundle: Path, tmp_path: Path
):
    expected_rows = 10
    rng = np.random.default_rng(99)
    data = {"feat_a": rng.random(expected_rows)}
    df = pd.DataFrame(data)
    missing_feat_csv = tmp_path/"missing_feat.csv"
    df.to_csv(missing_feat_csv, index=False)
    actual_pred_csv_path = tmp_path/"predictions.csv"

    run_inference(
        model_in=trained_model_bundle,
        data_csv=missing_feat_csv,
        target=None,
        exclude_cols=[],
        pred_csv=actual_pred_csv_path,
    )

    npt.assert_(actual_pred_csv_path.exists())
    actual_preds_df = pd.read_csv(actual_pred_csv_path)
    npt.assert_equal(len(actual_preds_df), expected_rows)
