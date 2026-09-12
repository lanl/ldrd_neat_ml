from typing import Optional

import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import logging
from neat_ml.model.train import ml_preprocess, plot_roc 

logger = logging.getLogger(__name__)


def save_predictions(
    df: pd.DataFrame, 
    y_prob: np.ndarray, 
    out_csv: Path
) -> None:
    """
    Calculate the predicted classifier labels and add them
    to the input dataframe of features along with the predicted
    probability values. Save the dataframe as a CSV file to the
    provided output path.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to save.
    y_prob : np.ndarray
        Positive-class probabilities.
    out_csv : Path
        Destination CSV path.
    """
    df_out = df.copy()
    df_out["Pred_Prob"] = y_prob
    # TODO: probability threshold (0.5) below is arbitrary
    # and should be updated to use the EER threshold (see issue #60)
    df_out["Pred_Label"] = (y_prob >= 0.5).astype(int)
    df_out.to_csv(out_csv, index=False)
    logger.info(f"Predictions saved -> {out_csv}")

def run_inference(
    model_in: Path,
    data_csv: Path,
    target: Optional[str],
    exclude_cols: list[str],
    roc_png: Path,
    pred_csv: Path,
) -> None:
    """
    Load a saved model bundle and use it to make classification
    predictions on inference data. Save the output predictions
    to a CSV file and generate a ROC plot if target labels are
    provided.

    Parameters
    ----------
    model_in : Path
        Path to the joblib bundle produced by training (must contain
        keys "model" and "features").
    data_csv : Path
        Path to the CSV file holding new data for inference.
    target : Optional[str]
        Name of the ground-truth column in `data_csv`.
        If provided a ROC plot will be generated.
    exclude_cols : list[str]
        List of column names to drop from `data_csv` before preprocessing.
    roc_png : Path
        File path to save the ROC plot.
    pred_csv : Path
        File path to save the output CSV of model predictions.
    """
    bundle = joblib.load(model_in)
    model = bundle["model"]
    feats = bundle["features"]

    df = pd.read_csv(data_csv)
    X, y = ml_preprocess(df, target, exclude_cols)

    X = X.reindex(columns=X.columns.union(feats))
    X = X[feats]

    y_prob = model.predict_proba(X)[:, 1]
    save_predictions(df, y_prob, pred_csv)
    if target is not None:
        plot_roc(y, y_prob, roc_png, label="Testing")
