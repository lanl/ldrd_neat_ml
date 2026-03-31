from typing import Sequence

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
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


__all__ = [
    "preprocess",
    "train_with_validation",
    "save_model_bundle",
]

def preprocess(
    df: pd.DataFrame,
    target: str,
    exclude: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
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
    exclude : list[str] | None
        A list of column names to exclude from 
        the feature set, by default None.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        A tuple containing the processed features 
        DataFrame (X) and the target Series (y).
    """
    y = pd.Series(dtype=int)
    mask = pd.Series(True, index=df.index)
    cols_to_drop = exclude if exclude else []

    if target:
        y = pd.to_numeric(df[target], errors="coerce")
        mask = y.notna()
        y = y[mask].astype(int)
        cols_to_drop.append(target)
    
    X = df.loc[mask].drop(columns=cols_to_drop, errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")

    # drop columns that are entirely NaN
    X = X.dropna(axis=1, how='all')

    X_imp = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X),
        columns=X.columns,
        index=X.index,
    )
    return X_imp, y


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
    n_jobs: int = -1,
    random_state: int = 42,
    ml_hyper_opt: bool = True,
) -> tuple[
    Pipeline,
    dict[str, float],
    dict[str, int | float | None],
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
    n_jobs : int
        The number of parallel processes to run
        when training the classifier. Default = -1
        aka use all available cores.
    random_state: int
        random seed variable for initializing
        machine learning classifiers
    ml_hyper_opt: bool
        whether or not to perform hyperparameter tuning
        via exhaustive grid search.

    Returns
    -------
    tuple[Pipeline, dict[str, float], 
    dict[str, int | float | None], np.ndarray]
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
    logger.info(f"scale_pos_weight={spw:.3f}  |  train neg/pos={np.bincount(y_train)}")

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        class_weight="balanced",
        n_jobs=n_jobs,
        random_state=random_state,
    )

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb)],
        voting="soft",
        n_jobs=n_jobs,
    )

    pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("ensemble", ensemble),
        ]
    )

    if ml_hyper_opt:
        param_grid = {
            # XGBoost
            "ensemble__xgb__n_estimators": [10, 20, 50, 100, 200, 400],
            "ensemble__xgb__learning_rate": [0.05, 0.1],
            "ensemble__xgb__max_depth": [3, 4, 5, 8, 10, 20],
        }

        X = np.concatenate((X_train, X_val), axis=0)
        y = np.concatenate((y_train, y_val), axis=0)

        train_val_split = [-1] * len(X_train) + [0] * len(X_val)
        ps = PredefinedSplit(train_val_split)
        
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=ps,
            scoring="roc_auc",
            n_jobs=n_jobs,
        )

        grid_search.fit(X, y)
        final_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
    else:
       final_model = pipeline.fit(X_train, y_train) 
       best_params = pipeline.get_params()

    val_proba = final_model.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_val, val_proba)
    best_score = roc_auc_score(y_val, val_proba)
    metrics = {"val_roc_auc": best_score, "val_pr_auc": pr_auc}
    
    logger.info(f"[BEST] val ROC-AUC={best_score:.4f}  | val PR-AUC={pr_auc:.4f}")
    logger.info(f"[BEST] hyper-parameters: {best_params}")

    return final_model, metrics, best_params, val_proba


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
    y_true : np.ndarray | Sequence[int]
        The true binary labels.
    y_prob : np.ndarray | Sequence[float]
        The predicted probabilities for the positive class.
    out_png : str
        The file path where the output PNG image will be saved.
    label : str, optional
        The label for the ROC curve in the plot legend, by 
        default "Validation".
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    aucv = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC={aucv:.3f}")
    ax.plot([0, 1], [0, 1], "--", lw=1, color="grey")
    ax.set_xlabel("False-Positive Rate")
    ax.set_ylabel("True-Positive Rate")
    ax.set_title(f"ROC Curve for {label} Dataset")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close()
    logger.info(f"ROC curve saved -> {out_png}")


def save_model_bundle(
    model: Pipeline,
    features: list[str],
    metrics: dict[str, float],
    best_params: dict[str, int | float | None],
    path: Path,
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
    features : list[str]
        The list of feature names used by the model.
    metrics : dict[str, float]
        A dictionary of performance metrics (e.g., {'val_roc_auc': 0.85}).
    best_params : dict[str, int | float | None]
        A dictionary of the best hyperparameters found during training.
    path : Path
        The file path where the model bundle will be saved.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": features,
            "metrics": metrics,
            "best_params": best_params,
        },
        path,
    )
    logger.info(f"Model bundle saved -> {path}")
