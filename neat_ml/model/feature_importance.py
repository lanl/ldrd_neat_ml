
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.feature_selection import SelectKBest

try:
    import shap
except ImportError:
    shap = None

try:
    from interpret.glassbox import ExplainableBoostingClassifier
except ImportError:
    ExplainableBoostingClassifier = None

try:
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:
    LimeTabularExplainer = None

warnings.filterwarnings("ignore", category=UserWarning)

__all__ = [
    "compare_methods",
    "feature_importance_consensus",
    "plot_feat_import_consensus",
]

RANDOM_STATE = 42

def _run_shap(
    model, X: pd.DataFrame, 
    out_dir: Path, 
    top: int = 20
) -> pd.Series:
    """
    Compute global SHAP values for *model* and derive per-feature importance.

    A permutation explainer is instantiated on the fly because it works with
    any black box predict_proba** function.  The absolute SHAP values are
    averaged across all rows, giving a single scalar importance per feature.

    Parameters
    ----------
    model : Any
        Fitted classifier exposing a predict_proba(X) -> ndarray method whose
        second dimension contains probabilities for the positive class.
    X : pandas.DataFrame
        Numeric feature matrix used both as background data for the explainer
        and as the evaluation set whose SHAP values are summarised.
    out_dir : pathlib.Path
        Directory where the SHAP bar chart (shap_summary.png) will be saved.
        The folder is assumed to exist.
    top : int, default 20
        Maximum number of features to display in the SHAP summary figure.

    Returns
    -------
    importance : pandas.Series
        Index = feature names, values = mean absolute SHAP value (descending).
        Empty series if the optional *shap* dependency is missing.
    """
    if shap is None:
        return pd.Series(dtype=float)
    explainer = shap.Explainer(
        model.predict_proba, masker=X, algorithm="permutation", n_jobs=-1
    )
    vals = explainer(X).values
    vals = vals[:, :, 1] if vals.ndim == 3 else vals
    imp = pd.Series(np.abs(vals).mean(0), index=X.columns).sort_values(ascending=False)

    shap.summary_plot(vals, features=X, max_display=top, show=False)
    plt.gcf().set_size_inches(8, 6)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=300)
    plt.close()
    return imp


def _run_ebm(
    X: pd.DataFrame,
    y: pd.Series,
    out_dir: Path,
    top: int = 20,
) -> pd.Series:
    """
    Train an ExplainableBoostingClassifier and extract main-effect importances.

    Parameters
    ----------
    X, y : pd.DataFrame, pd.Series
        Training data.
    out_dir : Path
        Folder for saving the EBM bar chart and CSV.
    top : int, default 20
        How many features appear in the bar chart.

    Returns
    -------
    pd.Series
        Importance values (descending). Empty if interpret not installed.
    """
    if ExplainableBoostingClassifier is None:  # pragma: no cover
        return pd.Series(dtype=float)

    ebm = ExplainableBoostingClassifier(
        interactions=0,
        random_state=RANDOM_STATE,
    ).fit(X, y)

    data = ebm.explain_global().data()
    names = np.asarray(data["names"])
    scores = np.asarray(data["scores"])

    mask_main = np.char.find(names, " * ") == -1 
    imp = pd.Series(np.abs(scores[mask_main]), index=names[mask_main])
    imp = imp.sort_values(ascending=False)
    plot_top = min(top, len(imp))
    imp.to_csv(out_dir / "ebm_importance.csv")
    plt.barh(np.arange(plot_top), imp.values[:plot_top])
    plt.yticks(np.arange(plot_top), imp.index[:plot_top])
    plt.tight_layout()
    plt.savefig(out_dir / "ebm_importance.png", dpi=300)
    plt.close()
    return imp

def _run_lime(
    model, X: pd.DataFrame, 
    n_samples: int = 100,
    *,
    random_state: int = RANDOM_STATE,
) -> pd.Series:
    """
    Estimate global feature importance via an aggregated LIME approach.

    Parameters
    ----------
    model : Any
        Fitted classifier with a probability interface compatible with LIME
        (predict_proba returning an *n_samples x 2 array for binary tasks).
    X : pandas.DataFrame
        Numeric feature matrix from which sampling is performed.
    n_samples : int, default 100
        Number of rows to sample for aggregation.  The function caps this value
        at len(X) to avoid redundant sampling on tiny datasets.

    Returns
    -------
    importance : pandas.Series
        Index = feature names, values = mean absolute LIME weight (descending).
        Empty series if the optional lime dependency is missing.

    """
    if LimeTabularExplainer is None:
        return pd.Series(dtype=float)

    rng = np.random.RandomState(random_state)  # o

    expl = LimeTabularExplainer(
        X.values,
        feature_names=X.columns.tolist(),
        class_names=["0", "1"],
        discretize_continuous=True,
        random_state=rng,
    )

    agg = np.zeros(X.shape[1])
    rows = rng.choice(len(X), min(n_samples, len(X)), replace=False)

    for i in rows:
        exp = expl.explain_instance(X.iloc[i].values, model.predict_proba,
                                    num_features=X.shape[1], labels=(1,))
        for f_idx, w in exp.as_map()[1]:
            agg[f_idx] += abs(w)
    return pd.Series(agg / len(rows), index=X.columns).sort_values(ascending=False)

def get_k_best_scores(
    X: pd.DataFrame,
    y: pd.Series,
    k: int,
    metrics: Sequence,
) -> list[np.ndarray]:
    """
    Run several *SelectKBest* filters and return their raw feature scores.

    Parameters
    ----------
    X, y : training data.
    k : int
        Number of top features each filter should select (used during fitting).
    metrics : Sequence
        Iterable of scoring callables from sklearn.feature_selection (e.g.
        f_classif, mutual_info_classif).

    Returns
    -------
    list[np.ndarray]
        Each array contains n_features scores for one metric.
    """
    scores: list[np.ndarray] = []
    for metric in metrics:
        sel = SelectKBest(metric, k=k).fit(X.to_numpy(), y)
        scores.append(sel.scores_)
    return scores

def feature_importance_consensus(
    pos_class_feat_imps: Sequence[npt.NDArray[np.float64]],
    feature_names: npt.NDArray[np.str_],
    top_feat_count: int,
) -> Tuple[npt.NDArray[np.str_], npt.NDArray[np.int64], int]:
    """
    Majority-vote ranking of the top_feat_count features across models.

    Parameters
    ----------
    pos_class_feat_imps : sequence of NumPy arrays
        Each array holds importances for one model (raw or per-sample SHAP).
    feature_names : 1-D array-like of str
        Feature name for each index.
    top_feat_count : int
        Top-k features extracted from every importance vector.

    Returns
    -------
    ranked_names : np.ndarray[str]
        Features sorted by descending consensus count.
    ranked_counts : np.ndarray[int]
        Occurrence counts corresponding to *ranked_names*.
    num_models : int
        Number of input importance sources.
    """
    num_models = len(pos_class_feat_imps)

    processed: list[np.ndarray] = []
    for arr in pos_class_feat_imps:
        if np.atleast_2d(arr).shape[0] > 1:
            processed.append(np.mean(np.abs(arr), axis=0))
        else:
            processed.append(np.abs(arr))

    votes: dict[str, int] = defaultdict(int)
    for imp in processed:
        top_idx = np.argsort(imp)[::-1][:top_feat_count]
        for name in feature_names[top_idx]:
            votes[str(name)] += 1

    votes = dict(sorted(votes.items(), key=lambda kv: kv[1], reverse=True))
    ranked_names = np.asarray(list(votes.keys()))
    ranked_counts = np.asarray(list(votes.values()), dtype=int)
    return ranked_names, ranked_counts, num_models


def plot_feat_import_consensus(
    ranked_names: npt.NDArray[np.str_],
    ranked_counts: npt.NDArray[np.int64],
    num_models: int,
    top_feat_count: int,
    out_dir: Path,
) -> None:
    """
    Horizontal bar chart of consensus occurrence x 100/num_models (%).

    Parameters
    ----------
    ranked_names, ranked_counts : outputs from feature_importance_consensus.
    num_models : int
        Denominator for % calculation.
    top_feat_count : int
        k used when building the consensus (only for axis label).
    out_dir: Path
        Location for the PNG file.

    Returns
    -------
    None
    """
    pct = (ranked_counts / num_models) * 100
    y_pos = np.arange(ranked_names.size)

    plt.figure(figsize=(6, 6))
    plt.barh(y_pos, pct)
    plt.gca().invert_yaxis()
    plt.xlabel(f"% models with feature in top {top_feat_count}")
    plt.yticks(y_pos, ranked_names.tolist())
    plt.title(f"Feature-importance consensus (n={num_models})")
    plt.tight_layout()
    plt.savefig(out_dir / "feat_imp_consensus.png", dpi=300),
    plt.close()
    print(f"Consensus plot saved -> {out_dir}")

def compare_methods(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    out_dir: Path,
    top: int = 20,
) -> None:
    """
    Run SHAP, EBM and LIME on *model* and merge their importances.

    The merged table is ranked by the **mean of method-specific ranks**
    (*lower* = more consistently important) before the top-k features are
    plotted.

    Parameters
    ----------
    model
        Fitted scikit-learn classifier with predict_proba.
    X, y : pd.DataFrame, pd.Series
        Data used for explanations.
    out_dir : Path
        Folder in which all artefacts will be written.
    top : int, default 20
        Number of features to display in the bar chart and to use for the
        consensus calculation.

    Returns
    -------
    None
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shap_imp = _run_shap(model, X, out, top=top)
    ebm_imp  = _run_ebm(X, y, out, top=top)
    lime_imp = _run_lime(model, X)

    feats = sorted(set(shap_imp.index) | set(ebm_imp.index) | set(lime_imp.index))
    comp = pd.DataFrame(index=feats)
    if not shap_imp.empty:
        comp["SHAP"] = shap_imp
    if not ebm_imp.empty:
        comp["EBM"] = ebm_imp
    if not lime_imp.empty:
        comp["LIME"] = lime_imp
    comp = comp.fillna(0)

    comp["mean_rank"] = comp.rank(ascending=False, method="average").mean(axis=1)
    comp = comp.sort_values("mean_rank")

    comp.to_csv(out / "feature_importance_comparison.csv")

    comp.head(top).drop(columns="mean_rank").plot.barh(figsize=(8, 6))
    plt.gca().invert_yaxis()
    plt.xlabel("importance (method-specific units)")
    plt.tight_layout()
    plt.savefig(out / "feature_importance_comparison.png", dpi=300)
    plt.close()

    if not comp.empty:
        ranked_names, ranked_counts, n_models = feature_importance_consensus(
            [comp[c].to_numpy(dtype=float) for c in ["SHAP", "EBM", "LIME"] if c in comp],
            comp.index.values.astype(str),
            top_feat_count=top,
        )
        plot_feat_import_consensus(
            ranked_names,
            ranked_counts,
            n_models,
            top,
            out,
        )