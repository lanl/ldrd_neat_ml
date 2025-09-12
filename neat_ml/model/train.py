import os
from itertools import product
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE: int = 42

_PARAM_GRID: Dict[str, List[int | float | None]] = {
    # Random‑Forest
    "ensemble__rf__n_estimators": [500],
    "ensemble__rf__max_depth": [None],
    # XGBoost
    "ensemble__xgb__n_estimators": [10, 20, 50, 100, 200, 400],
    "ensemble__xgb__learning_rate": [0.05, 0.1],
    "ensemble__xgb__max_depth": [3, 4, 5, 8, 10, 20],
}

__all__ = [
    "preprocess",
    "train_with_validation",
    "save_model_bundle",
]

def preprocess(
    df: pd.DataFrame,
    target: str,
    exclude: Sequence[str] | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target, and preprocess 
    the data for modeling.

    This function takes a raw DataFrame, separates
    the features from the target variable, and performs
    initial cleaning. Rows with missing target values
    are dropped. All feature columns are converted 
    to numeric types, and any remaining missing values 
    in the features are imputed using the median.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing features and 
        the target variable.
    target : str
        The name of the target column.
    exclude : Sequence[str] | None, optional
        A sequence of column names to exclude from 
        the feature set, by default None.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        A tuple containing the processed features 
        DataFrame (X) and the target Series (y).
    """
    df = df.copy()
    y = pd.Series(dtype=int)
    mask = pd.Series(True, index=df.index)
    cols_to_drop = list(exclude) if exclude else []

    if target:
        y = pd.to_numeric(df[target], errors="coerce")
        mask = y.notna()
        y = y[mask].astype(int)
        cols_to_drop.append(target)
    
    X = df.loc[mask].drop(columns=cols_to_drop, errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")

    # Add this line to drop columns that are entirely NaN
    X = X.dropna(axis=1, how='all')

    X_imp = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X),
        columns=X.columns,
        index=X.index,
    )
    return X_imp, y


def _build_pipeline(
    scale_pos_weight: float
) -> Pipeline:
    """
    Build a scikit-learn pipeline with an RF + XGB 
    soft-voting ensemble.

    The pipeline consists of three steps:
    1. Median imputation for missing values.
    2. Feature scaling using StandardScaler.
    3. A soft-voting ensemble of RandomForestClassifier
       and XGBClassifier.

    Parameters
    ----------
    scale_pos_weight : float
        The weight for the positive class, used by XGBoost
        to handle class imbalance.

    Returns
    -------
    Pipeline
        The configured scikit-learn pipeline.
    """
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb)],
        voting="soft",
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ensemble", ensemble),
        ]
    )


def _scale_pos_weight(y: pd.Series) -> float:
    """
    Calculate the negative/positive class ratio 
    for XGBoost's `scale_pos_weight`.

    Parameters
    ----------
    y : pd.Series
        The target variable Series containing 
        binary class labels.

    Returns
    -------
    float
        The calculated ratio of negative to positive 
        samples. Returns 1.0 if the positive class 
        is not present.
    """
    counts = np.bincount(y, minlength=2)
    neg, pos = counts
    return neg / pos if pos else 1.0


def train_with_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Tuple[
    Pipeline,
    Dict[str, float],
    Dict[str, int | float | None],
    np.ndarray,
]:
    """
    Train an ensemble classifier with hyperparameter 
    tuning on a validation set.

    This function iterates over a predefined 
    hyperparameter grid, fitting each candidate model 
    on the training set and evaluating its ROC-AUC 
    score on the validation set. The parameters of 
    the best model are then used to train a final model
    on the combined training and validation data.

    Parameters
    ----------
    X_train : pd.DataFrame
        The feature matrix for the training set.
    y_train : pd.Series
        The target vector for the training set.
    X_val : pd.DataFrame
        The feature matrix for the validation set.
    y_val : pd.Series
        The target vector for the validation set.

    Returns
    -------
    Tuple[Pipeline, Dict[str, float], 
    Dict[str, int | float | None], np.ndarray]
        A tuple containing:
        - The final model pipeline, refit on 
          the combined train+validation data.
        - A dictionary of performance metrics 
          (ROC-AUC, PR-AUC) on the validation set.
        - A dictionary of the best hyperparameters
          found during the grid search.
        - The predicted probabilities for the positive
          class on the validation set.
    """
    spw = _scale_pos_weight(y_train)
    print(f"[INFO] scale_pos_weight={spw:.3f}  |  train neg/pos={np.bincount(y_train)}")

    param_names = list(_PARAM_GRID.keys())
    param_values = list(_PARAM_GRID.values())
    best_auc = -np.inf
    best_params: Dict[str, int | float | None] = {}
    best_model: Pipeline | None = None
    val_proba_best = None

    for param_comb in product(*param_values):
        params = dict(zip(param_names, param_comb))
        model = _build_pipeline(spw).set_params(**params)

        model.fit(X_train, y_train)
        val_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_proba)
        print(f"[GRID] params={params}  |  val ROC-AUC={auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_params = params
            best_model = model
            val_proba_best = val_proba

    assert best_model is not None, "No model was fitted!"
    assert val_proba_best is not None, "Validation probabilities were not calculated!"

    pr_auc = average_precision_score(y_val, val_proba_best)
    metrics = {"val_roc_auc": best_auc, "val_pr_auc": pr_auc}

    print(f"[BEST] val ROC-AUC={best_auc:.4f}  | val PR-AUC={pr_auc:.4f}")
    print(f"[BEST] hyper-parameters: {best_params}")

    X_full = pd.concat([X_train, X_val], axis=0)
    y_full = pd.concat([y_train, y_val], axis=0)
    final_model = _build_pipeline(
        _scale_pos_weight(y_full)
        ).set_params(**best_params)
    final_model.fit(X_full, y_full)

    return final_model, metrics, best_params, val_proba_best


def plot_roc(
    y_true: np.ndarray | Sequence[int],
    y_prob: np.ndarray | Sequence[float],
    out_png: str,
    label: str = "Validation",
) -> None:
    """
    Generate and save a Receiver Operating Characteristic
    (ROC) curve plot.

    Calculates and plots the ROC curve and the Area Under
    the Curve (AUC) score, then saves the figure to a PNG file.

    Parameters
    ----------
    y_true : Sequence[int]
        The true binary labels.
    y_prob : Sequence[float]
        The predicted probabilities for the positive class.
    out_png : str
        The file path where the output PNG image will be saved.
    label : str, optional
        The label for the ROC curve in the plot legend, by 
        default "Validation".
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    aucv = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, lw=2, label=f"{label} AUC={aucv:.3f}")
    plt.plot([0, 1], [0, 1], "--", lw=1, color="grey")
    plt.xlabel("False-Positive Rate")
    plt.ylabel("True-Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[INFO] ROC curve saved -> {out_png}")


def save_model_bundle(
    model: Pipeline,
    features: List[str],
    metrics: Dict[str, float],
    best_params: Dict[str, int | float | None],
    path: str,
) -> None:
    """
    Serialize and save the trained model and associated 
    metadata to a file.

    This function bundles the trained pipeline, feature 
    list, performance metrics, and best hyperparameters 
    into a dictionary and saves it using joblib.

    Parameters
    ----------
    model : Pipeline
        The trained scikit-learn pipeline to be saved.
    features : List[str]
        The list of feature names used by the model.
    metrics : Dict[str, float]
        A dictionary of performance metrics (e.g., {'val_roc_auc': 0.85}).
    best_params : Dict[str, int | float | None]
        A dictionary of the best hyperparameters found during training.
    path : str
        The file path where the model bundle will be saved.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": features,
            "metrics": metrics,
            "best_params": best_params,
        },
        path,
    )
    print(f"[INFO] Model bundle saved -> {path}")