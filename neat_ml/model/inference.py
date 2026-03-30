from typing import Optional

import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import logging
from .train import preprocess, plot_roc

logger = logging.getLogger(__name__)


def save_predictions(
    df: pd.DataFrame, 
    y_prob: np.ndarray, 
    out_csv: Path
) -> None:
    """
    Append probability & label columns and write CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Source data (+ optional ground truth).
    y_prob : np.ndarray
        Positive-class probabilities.
    out_csv : Path
        Destination CSV path.
    """
    df = df.copy()
    df["Pred_Prob"] = y_prob
    df["Pred_Label"] = (y_prob >= 0.5).astype(int)
    df.to_csv(out_csv, index=False)
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
    Load a saved model bundle, run it on new data, and emit both
    a predictions CSV and (if labels are provided) a ROC curve PNG.

    Parameters
    ----------
    model_in : Path
        Path to the joblib bundle produced by training (must contain
        keys "model" and "features").
    data_csv : Path
        Path to the CSV file holding new data for inference.
    target : Optional[str]
        Name of the ground-truth column in `data_csv`. If given and
        non-empty, a ROC curve will be generated.
    exclude_cols : list[str]
        List of column names to drop from `data_csv` before preprocessing.
    roc_png : Path
        File path where the ROC curve image (PNG) will be saved.
    pred_csv : Path
        File path where the output CSV of predictions will be written.
    """
    bundle = joblib.load(model_in)
    model = bundle["model"]
    feats = bundle.get("features", [])

    df = pd.read_csv(data_csv)
    X, y = preprocess(df, target or "", exclude_cols)

    X = X.reindex(columns=list(set(X.columns) | set(feats)))
    X = X[feats] if feats else X

    y_prob = model.predict_proba(X)[:, 1]
    save_predictions(df, y_prob, pred_csv)
    if target is not None:
        plot_roc(y.to_list(), y_prob, str(roc_png), label="Testing")
