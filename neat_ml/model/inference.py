from typing import Sequence, Optional

import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from .train import preprocess

def plot_roc(
    y_true: np.ndarray | Sequence[int],
    y_prob: np.ndarray | Sequence[float],
    out_path: Path,
    out_png: Path
) -> None:
    """
    Save a ROC curve PNG.

    Parameters
    ----------
    y_true : Sequence[int]
        Ground-truth binary labels.
    y_prob : Sequence[float]
        Predicted positive probabilities.
    out_png : Path
        Output file path.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    aucv = roc_auc_score(y_true, y_prob)
    out_path.mkdir(parents=True, exist_ok=True)
    output_roc_image = out_path / out_png
    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, label=f"AUC={aucv:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_roc_image, dpi=300)
    plt.close()
    print(f"[INFO] ROC saved -> {output_roc_image}")


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
    print(f"[INFO] Predictions saved -> {out_csv}")

def run_inference(
    model_in: Path,
    data_csv: Path,
    target: Optional[str],
    exclude_cols: list[str],
    roc_out: Path,
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
    roc_out : Path
        File path directory where the ROC curve image (PNG) will be saved.
    roc_png : Path
        File path where the ROC curve image (PNG) will be saved.
    pred_csv : Path
        File path where the output CSV of predictions will be written.

    Returns
    -------
    None
    """
    bundle = joblib.load(model_in)
    model = bundle["model"]
    feats = bundle.get("features", [])

    df = pd.read_csv(data_csv)
    X, y = preprocess(df, target or "", exclude_cols)

    for col in feats:
        if col not in X.columns:
            X[col] = np.nan
    X = X[feats] if feats else X

    y_prob = model.predict_proba(X)[:, 1]
    save_predictions(df, y_prob, pred_csv)

    if target and not y.empty:
        plot_roc(y.tolist(), y_prob, roc_out, roc_png)
