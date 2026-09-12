import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal, assert_allclose

from neat_ml.model.inference import run_inference, save_predictions


def test_save_predictions(tmp_path):
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
    
    # assert that the input_df is not mutated by the function
    assert_array_equal(input_df.columns, ["id", "data"])
    actual_df = pd.read_csv(actual_out_csv)
    # assert that `Pred_Prob` was added to the input df 
    assert_allclose(actual_df["Pred_Prob"], pred_prob)
    # assert that predicted labels were calculated correctly
    # based on the predicted probability of the ML classifier
    assert_array_equal(actual_df["Pred_Label"], [0, 1, 1])
    # assert that the output df still contains the columns
    # from the input df
    assert_array_equal(actual_df[["id", "data"]], input_df)

@pytest.mark.parametrize("target, roc_plot",
    [
        ("ground_truth", "roc.png"),
        (None, None),
    ]
)
def test_run_inference(
    trained_model_bundle,
    sample_inference_data,
    tmp_path,
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
    # check a subset of expected output probability values/predictions
    assert_allclose(
        actual_preds_df["Pred_Prob"].iloc[:5],
        np.array([0.27, 0.21, 0.33, 0.33, 0.33])
    ) 
    assert_array_equal(actual_preds_df["Pred_Label"].iloc[:5], [0] * 5)
    if target is not None:
        assert (tmp_path / "roc.png").exists()


def test_run_inference_handles_missing_feature(trained_model_bundle, tmp_path):
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
