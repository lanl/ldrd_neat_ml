from collections import defaultdict
from pathlib import Path
from typing import Sequence, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest

import shap
from interpret.glassbox import ExplainableBoostingClassifier
from lime.lime_tabular import LimeTabularExplainer
from sklearn.pipeline import Pipeline


__all__ = [
    "compare_methods",
    "feature_importance_consensus",
    "plot_feat_import_consensus",
]


def _run_shap(
    model, X: pd.DataFrame, 
    out_dir: Path, 
    top: int = 20,
    n_jobs: int = -1,
    rng: np.random.Generator | None = None,
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
        and as the evaluation set whose SHAP values are summarized.
    out_dir : pathlib.Path
        Directory where the SHAP bar chart (shap_summary.png) will be saved.
    top : int, default 20
        Maximum number of features to display in the SHAP summary figure.
    n_jobs : int
        number of parallel processes to run for shap explainer. n_jobs=-1 uses
        all cores.
    rng : np.random.Generator | None
        pseudorandom number generator

    Returns
    -------
    imp : pandas.Series
        Index = feature names, values = mean absolute SHAP value (descending).
    """
    explainer = shap.Explainer(
        model.predict_proba,
        masker=X.values,
        algorithm="permutation",
        n_jobs=n_jobs,
        feature_names=X.columns.to_list(),
    )
    vals = explainer(X.values).values
    vals = vals[:, :, 1] if vals.ndim == 3 else vals
    imp = pd.Series(np.abs(vals).mean(0), index=X.columns).sort_values(ascending=False)

    shap.summary_plot(vals, features=X, max_display=top, show=False, rng=rng)
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
    random_state: int = 42
) -> pd.Series:
    """
    Train an ExplainableBoostingClassifier and extract main-effect importances.

    Parameters
    ----------
    X : pd.DataFrame
        Input training data for ExplainableBoostingClassifier.
    y : pd.Series
        Training targets
    out_dir : Path
        Folder for saving the EBM bar chart and CSV.
    top : int, default 20
        Maximum number of features to display in the SHAP summary figure.
    random_state: int
        random seed variable for initializing explainer

    Returns
    -------
    imp : pd.Series
        Importance values (descending).
    """
    ebm = ExplainableBoostingClassifier(
        interactions=0,
        random_state=random_state,
        feature_names=X.columns.to_list()
    ).fit(X.values, y)

    data = ebm.explain_global().data()
    names = np.asarray(data["names"])
    scores = np.asarray(data["scores"])

    mask_main = np.char.find(names, " * ") == -1 
    imp = pd.Series(np.abs(scores[mask_main]), index=names[mask_main])
    imp = imp.sort_values(ascending=False)
    plot_top = min(top, len(imp))
    imp.to_csv(out_dir / "ebm_importance.csv")
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.barh(np.arange(plot_top), imp.values[:plot_top])  # type: ignore[arg-type]
    ax.set_yticks(np.arange(plot_top), imp.index[:plot_top])  # type: ignore[arg-type]
    fig.tight_layout()
    fig.savefig(out_dir / "ebm_importance.png", dpi=300)
    plt.close(fig)
    return imp

def _run_lime(
    model: Pipeline,
    X: pd.DataFrame, 
    n_samples: int = 100,
    *,
    random_state: int = 42,
) -> pd.Series:
    """
    Estimate global feature importance via an aggregated LIME approach.

    Parameters
    ----------
    model : Pipeline
        Fitted classifier with a probability interface compatible with LIME
        (predict_proba returning an n_samples x 2 array for binary tasks).
    X : pandas.DataFrame
        Numeric feature matrix from which sampling is performed.
    n_samples : int, default 100
        Number of rows to sample for aggregation.  The function caps this value
        at len(X) to avoid redundant sampling on tiny datasets.
    random_state: int
        random seed variable for initializing LIME explainer

    Returns
    -------
    pandas.Series
        Index = feature names, values = mean absolute LIME weight (descending).
    """
    rng = np.random.default_rng(random_state)

    expl = LimeTabularExplainer(
        X.values,
        feature_names=X.columns.tolist(),
        class_names=["0", "1"],
        discretize_continuous=True,
        random_state=random_state,
    )

    agg = np.zeros(X.shape[1])
    # select random rows on which to train the LIME explainer because
    # training on all rows is computationally expensive
    rows = rng.choice(len(X), min(n_samples, len(X)), replace=False)

    for i in rows:
        # for each row, generate LIME explanation
        exp = expl.explain_instance(X.iloc[i].values, model.predict_proba,
                                    num_features=X.shape[1], labels=(1,))
        # get feature indices and weights for the explanation and
        # aggregate the importances for each feature
        for f_idx, w in exp.as_map()[1]:
            agg[f_idx] += abs(w)
    # return the sorted average absolute importance per feature        
    return pd.Series(agg / len(rows), index=X.columns).sort_values(ascending=False)

def get_k_best_scores(
    X: pd.DataFrame,
    y: pd.Series,
    k: int,
    metrics: Sequence,
) -> list[np.ndarray]:
    """
    Run several SelectKBest filters and return their raw feature scores.

    Parameters
    ----------
    X : pd.DataFrame
        dataframe containing training data
    y : pd.Series
        pandas series containing training targets
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
    scores = []
    for metric in metrics:
        sel = SelectKBest(metric, k=k).fit(X, y)
        scores.append(sel.scores_)
    return scores

def feature_importance_consensus(
    pos_class_feat_imps: Sequence[np.ndarray[Any, np.dtype[np.float64]]],
    feature_names: np.ndarray[Any, np.dtype[np.str_]],
    top_feat_count: int,
) -> tuple[np.ndarray[Any, np.dtype[np.str_]], np.ndarray[Any, np.dtype[np.int64]], int]:
    """
    Majority-vote ranking of the top_feat_count features across models.

    Parameters
    ----------
    pos_class_feat_imps : Sequence[np.ndarray]
        Each array holds importances for one model.
    feature_names : np.ndarray[str]
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

    processed = []
    for arr in pos_class_feat_imps:
        if np.atleast_2d(arr).shape[0] > 1:
            processed.append(np.mean(np.abs(arr), axis=0))
        else:
            processed.append(np.abs(arr))

    votes = defaultdict(int) # type: ignore[var-annotated]
    for imp in processed:
        top_idx = np.argsort(imp)[::-1][:top_feat_count]
        for name in feature_names[top_idx]:
            votes[name] += 1

    votes = dict(sorted(votes.items(), key=lambda kv: kv[1], reverse=True)) # type: ignore[assignment]
    ranked_names = np.asarray(list(votes.keys()))
    ranked_counts = np.asarray(list(votes.values()), dtype=int)
    return ranked_names, ranked_counts, num_models

def plot_feat_import_consensus(
    ranked_names: np.ndarray[Any, np.dtype[np.str_]],
    ranked_counts: np.ndarray[Any, np.dtype[np.int_]],
    num_models: int,
    top_feat_count: int,
    out_dir: Path,
) -> None:
    """
    Horizontal bar chart of consensus occurrence x 100/num_models (%).

    Parameters
    ----------
    ranked_names : np.ndarray
        ranked feature names sorted by descending consensus count
    ranked_counts : np.ndarray
        feature counts corresponding to ``ranked_names``
    num_models : int
        Total number of models used in consensus.
        Denominator for % calculation.
    top_feat_count : int
        k used when building the consensus (only for axis label).
    out_dir: Path
        Location for saving the PNG file.
    """
    pct = (ranked_counts / num_models) * 100
    y_pos = np.arange(ranked_names.size)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.barh(y_pos, pct)
    ax.invert_yaxis()
    ax.set_xlabel(f"% models with feature in top {top_feat_count}")
    ax.set_yticks(y_pos, ranked_names.tolist())
    ax.set_title(f"Feature-importance consensus (n={num_models})")
    fig.tight_layout()
    fig.savefig(out_dir / "feat_imp_consensus.png", dpi=300)
    plt.close()
    print(f"Consensus plot saved -> {out_dir}")

def compare_methods(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    out_dir: Path,
    top: int = 20,
    rng: np.random.Generator | None = None,
) -> None:
    """
    Run SHAP, EBM and LIME on *model* and merge their importances.

    The merged table is ranked by the **mean of method-specific ranks**
    (*lower* = more consistently important) before the top-k features are
    plotted.

    Parameters
    ----------
    model: Pipeline
        Fitted scikit-learn classifier with predict_proba.
    X : pd.DataFrame
        training data for performing feature importance explanation
    y : pd.Series
        training data targets
    out_dir : Path
        Folder in which all artifacts will be written.
    top : int, default 20
        Number of features to display in the bar chart and to use for the
        consensus calculation.
    rng : np.random.Generator
        pseudorandom number generator
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    shap_imp = _run_shap(model, X, out_dir, top=top, rng=rng)
    ebm_imp  = _run_ebm(X, y, out_dir, top=top)
    lime_imp = _run_lime(model, X)

    feats = sorted(set(shap_imp.index) | set(ebm_imp.index) | set(lime_imp.index))
    comp = pd.DataFrame(index=feats)
    comp["SHAP"] = shap_imp
    comp["EBM"] = ebm_imp
    comp["LIME"] = lime_imp
    comp = comp.fillna(0)

    comp["mean_rank"] = comp.rank(ascending=False, method="average").mean(axis=1)
    comp = comp.sort_values("mean_rank")

    comp.to_csv(out_dir / "feature_importance_comparison.csv")
    
    plot_feature_importance_comparison(comp, out_dir, top)

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
        out_dir,
    )

def plot_feature_importance_comparison(
    comp: pd.DataFrame,
    out_dir: Path,
    top: int,
) -> None:
    """
    plot the comparison of the three feature importance methods
    on a single bar plot using the output csv from calling ``compare_methods``

    Parameters:
    -----------
    comp: pd.DataFrame
        dataframe holding values for feature importance comparison
    out_dir: Path
        path to save feature importance comparison plot
    top: int
        top n features to plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    comp.head(top).drop(columns="mean_rank").plot.barh(ax=ax)
    ax.invert_yaxis()
    ax.set_xlabel("Importance Value (method-specific units)")
    ax.set_title("Comparison of Feature Importance Rankings")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance_comparison.png", dpi=300)
    plt.close()
