from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from pandas.testing import assert_frame_equal, assert_index_equal

from neat_ml.model.inference import run_inference, save_predictions


def test_save_predictions(tmp_path: Path):
    input_df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "data": ["a", "b", "c"],
        }
    )

    pred_prob = np.array([0.1, 0.5, 0.9])
    actual_out_csv = tmp_path / "preds.csv"

    save_predictions(
        input_df,
        pred_prob,
        actual_out_csv
    )

    assert_index_equal(input_df.columns, pd.Index(["id", "data"]))
    actual_df = pd.read_csv(actual_out_csv)
    assert_allclose(actual_df["Pred_Prob"], pred_prob)
    assert_equal(actual_df["Pred_Label"].to_list(), [0, 1, 1])
    assert_frame_equal(actual_df[["id", "data"]], input_df)

@pytest.mark.parametrize("target, roc_plot",
    [
        ("ground_truth", "roc.png"),
        (None, None),
    ]
)
def test_run_inference(
    trained_model_bundle: Path,
    sample_inference_data: Path,
    tmp_path: Path,
    target,
    roc_plot
):
    """
    test that ``run_inference`` generates the appropriate csv file for
    each input and that when a target variable is provided, it also
    produces a ROC plot.
    """
    actual_pred_csv_path = tmp_path / "predictions.csv"

    run_inference(
        model_in=trained_model_bundle,
        data_csv=sample_inference_data,
        target=target,
        exclude_cols=["id_col"],
        roc_png=tmp_path / "roc.png",
        pred_csv=actual_pred_csv_path,
    )

    actual_preds_df = pd.read_csv(actual_pred_csv_path)
    assert len(actual_preds_df) == 50
    # check a subset of expected output probability values
    assert_allclose(actual_preds_df["Pred_Prob"].iloc[:5], np.array([0.27, 0.21, 0.33, 0.33, 0.33])) 
    assert set(["Pred_Prob", "Pred_Label"]).issubset(actual_preds_df.columns)
    if target is not None:
        assert (tmp_path / "roc.png").exists()


def test_run_inference_handles_missing_feature(
    trained_model_bundle: Path, tmp_path: Path
):
    """
    test that the model predictions are still made
    even when one of the training feature columns
    is missing from the inference dataset
    """
    expected_rows = 10
    rng = np.random.default_rng(99)
    data = {"feat_a": rng.random(expected_rows)}
    df = pd.DataFrame(data)
    missing_feat_csv = tmp_path / "missing_feat.csv"
    df.to_csv(missing_feat_csv, index=False)
    actual_pred_csv_path = tmp_path / "predictions.csv"

    run_inference(
        model_in=trained_model_bundle,
        data_csv=missing_feat_csv,
        target=None,
        exclude_cols=[],
        roc_png=tmp_path / "roc.png",
        pred_csv=actual_pred_csv_path,
    )

    actual_preds_df = pd.read_csv(actual_pred_csv_path)
    assert len(actual_preds_df) == expected_rows
    assert_allclose(
        actual_preds_df["Pred_Prob"],
        np.array([0.09, 0.07, 0.09, 0.08, 0.07, 0.07, 0.48, 0.07, 0.09, 0.07])
    ) 
    assert_array_equal(
        actual_preds_df.columns,
        ["feat_a", "Pred_Prob", "Pred_Label"]
    )
