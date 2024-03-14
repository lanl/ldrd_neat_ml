import pickle
from pathlib import Path
import os
import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import minimize
import pandas as pd
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, GridSearchCV,
                                     cross_val_predict)
from sklearn.ensemble import (RandomForestClassifier, StackingClassifier,
                              VotingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_array
import xgboost as xgb
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import shap
from interpret.glassbox import ExplainableBoostingClassifier
import lightgbm as lgb


from neat_ml import lib


def main():
    # Step 1: Read in the data/format it appropriately
    # Mihee's experimental PEO/DEX binary phase sep data:
    df = pd.read_excel("neat_ml/data/mihee_peo_dextran_phase_map_experimental.xlsx")
    # shape (34, 4)
    X, y = lib.preprocess_data(df=df)
    # read in Cesar's CG/AA-MD simulation data for PEO/DEX:
    df_cesar_cg = lib.read_in_cesar_cg_md_data() # shape (49, 911)
    df_cesar_aa = lib.read_in_cesar_all_atom_md_data() # shape (49, 18)
    df_cesar_combined = lib._merge_dfs(df_cesar_cg, df_cesar_aa)
    assert df_cesar_combined.shape == (49, 911 + 18 - 3)
    # some of the columns are apparently just constant values,
    # so filter those out (they can't possibly contribute to
    # ML target selection)
    col_inds_where_constant = np.argwhere(np.diff(df_cesar_combined, axis=0).sum(axis=0) == 0).ravel()
    constant_cols = df_cesar_combined.columns[col_inds_where_constant]
    print("filtering out constant MD data colums:", constant_cols)
    df_cesar_combined.drop(labels=constant_cols,
                           axis="columns",
                           inplace=True)
    check_array(df_cesar_combined)

    # Plot the experimental vs. CG/AA-MD input PEO/Dextran maps
    # so we get an idea of the phase space we're comparing
    # (they are a bit different, but mostly overlap, as intended)
    lib.plot_input_data(X, y)
    lib.plot_input_data_cesar_MD(df=df_cesar_combined)

    # Step 1b: also plot triangle phase diagram
    # TODO: use actual 3-species/polymer data--for now we just
    # use synthetic data for block copolymer to check that
    # we produce something reasonable
    y_tmp = 50 - X.sum(axis=1)
    X_tmp = np.column_stack((X[..., 0], X[..., 1], y_tmp))
    lib.plot_tri_phase_diagram(X_tmp, y, plot_path=os.getcwd())

    # we only have two features at the moment (% Dextran, % PEO)
    # so no need for feature selection just yet; can jump right into
    # hyperparameter optimization

    # Step 2: Split the data into training and test sets
    # we don't have much data for now, but we still need
    # to split into training and test sets to do any kind
    # of useful ML...
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # 2b: plot/output the class imbalance between training/test sets
    training_no_phase_sep_count = (y_train == 0).sum()
    training_phase_sep_count = (y_train == 1).sum()
    test_no_phase_sep_count = (y_test == 0).sum()
    test_phase_sep_count = (y_test == 1).sum()
    fig_class_imb, ax = plt.subplots()
    ax.bar(["training NO",
            "training YES",
            "test NO",
            "test YES"],
           height=[training_no_phase_sep_count,
                   training_phase_sep_count,
                   test_no_phase_sep_count,
                   test_phase_sep_count],
           color=["blue", "blue", "red", "red"])
    ax.set_xlabel("Phase Separation?")
    ax.set_ylabel("Record Count")
    training_percent_phase_sep = (training_phase_sep_count / y_train.size) * 100.
    test_percent_phase_sep = (test_phase_sep_count / y_test.size) * 100.
    print(f"% phase separated training: {training_percent_phase_sep:.2f}")
    print(f"% phase separated test: {test_percent_phase_sep:.2f}")
    ax.set_title(f"Binary Phase Separation Class Imbalance (Train: {training_percent_phase_sep:.2f} %; Test: {test_percent_phase_sep:.2f} %)")
    fig_class_imb.savefig("class_imbalance.png", dpi=300)

    # Step 3: Establish baseline cross-validation scores on training
    # let's establish baseline cross-validation roc_auc scores
    # prior to any hyperparameter optimization, using only
    # the training data

    estimator_data = {
                      # standard sklearn random forest:
                      "rfc": {"classifier": RandomForestClassifier(random_state=42)},
                      # standard xgboost classifier:
                      "xgb_class": {"classifier": xgb.XGBClassifier()},
                      # xgboost dropout classifier:
                      "xgb_dart": {"classifier": xgb.XGBClassifier(booster="dart",
                                                                   one_drop=1)},
                      # sklearn SVM:
                      "svm": {"classifier": SVC(gamma='auto', probability=True)},
                      }

    for estimator_name in estimator_data.keys():
        estimator = estimator_data[estimator_name]["classifier"]
        if estimator_name == "svm":
            # need the scaler for SVM
            estimator = make_pipeline(StandardScaler(), estimator)
            # TODO: safety check per:
            # https://stackoverflow.com/a/43366811/2942522
        scores = cross_val_score(estimator,
                                 X_train,
                                 y_train,
                                 scoring="roc_auc",
                                 cv=StratifiedKFold(5))
        estimator_data[estimator_name]["baseline_average_auc"] = np.average(scores)

    print("-" * 70)
    print("baseline estimator average CV ROC AUC score data on training only:")
    for estimator_name in estimator_data.keys():
        print(f"estimator_name: {estimator_name},",
              "average AUC:",
              estimator_data[estimator_name]["baseline_average_auc"]
              )
    print("-" * 70)

    # Step 4: Perform some basic hyperparameter searching/optimization
    # NOTE: at the moment, we have so little data that this optimization
    # is really just here as a skeleton for a larger data/workflow in the future
    # When we need to handle a larger search space, we may need to use i.e.,
    # RandomizedSearchCV as a compromise over exhaustive grid searching.
    print("-" * 70)
    print("Hyperparameter optimization on training only:")
    for estimator_name in estimator_data.keys():
        hyperparam_input = lib.hyper_param_dict[estimator_name]
        cls = estimator_data[estimator_name]["classifier"]
        # TODO: scaler for SVM/pipeline handling
        # careful: https://stackoverflow.com/a/43366811/2942522
        clf = GridSearchCV(cls,
                           hyperparam_input)
        clf.fit(X_train, y_train)
        print(f"setting best params for {estimator_name} estimator:", clf.best_params_)
        estimator = estimator_data[estimator_name]["classifier"]
        estimator.set_params(**clf.best_params_)
        if estimator_name == "svm":
            # need the scaler for SVM
            estimator = make_pipeline(StandardScaler(), estimator)
        scores = cross_val_score(estimator,
                                 X_train,
                                 y_train,
                                 scoring="roc_auc",
                                 cv=StratifiedKFold(5))
        new_score = np.average(scores)
        estimator_data[estimator_name]["hyperparam_average_auc"] = new_score
    for estimator_name in estimator_data.keys():
        print(f"estimator_name: {estimator_name},",
              "average AUC:",
              estimator_data[estimator_name]["hyperparam_average_auc"]
              )
    print("-" * 70)

    # Step 5: Assess estimator orthogonality so we get a sense for which
    # estimators may be suitably combined via various forms of ensembling
    # (i.e., soft voting, stacking, etc.) to improve our final classifier
    # performance on test
    print("-" * 70)
    print("Assessment of estimator orthogonality based on "
          "Pearson correlation coefficient\nof CV-predictions "
          "on training data.\nPurpose is to assess suitability "
          "for eventual ensembling on test.")
    ortho_data = {}
    for estimator_name in estimator_data.keys():
        estimator = estimator_data[estimator_name]["classifier"]
        if estimator_name == "svm":
            # need the scaler for SVM
            estimator = make_pipeline(StandardScaler(), estimator)

        # generate cross validated estimates for each training data point
        pred_proba = cross_val_predict(estimator=estimator,
                                       X=X_train,
                                       y=y_train,
                                       cv=StratifiedKFold(5),
                                       method="predict_proba")
        # the second column of the pred_proba array should
        # be the probabilities for phase separation ("1" value)
        pred_proba = pred_proba[..., 1]
        ortho_data[estimator_name] = pred_proba
    df_ortho = pd.DataFrame.from_dict(ortho_data).corr(method="pearson")
    print(df_ortho)
    df_ortho.style.pipe(lib.color_df).to_html("estimator_orthogonality_training.html")
    print("-" * 70)

    # Step 6: train each estimator on the full training data, then
    # produce ROC curves for each on test
    for estimator_name in estimator_data.keys():
        estimator = estimator_data[estimator_name]["classifier"]
        if estimator_name == "svm":
            # need the scaler for SVM
            estimator = make_pipeline(StandardScaler(), estimator)
        estimator.fit(X_train, y_train)
        pred_proba = estimator.predict_proba(X_test)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_proba[..., 1])
        roc_auc = metrics.auc(fpr, tpr)
        roc_plot = metrics.RocCurveDisplay(fpr=fpr,
                                           tpr=tpr,
                                           roc_auc=roc_auc,
                                           estimator_name=f"{estimator_name}")
        roc_plot.plot()
        fig = roc_plot.figure_
        fig.savefig(f"roc_{estimator_name}.png", dpi=300)

    # Step 7: Ensembling to improve predictions
    # From the orthogonality check above, at the time of writing,
    # if we assume that the xgb classifier is our "base classifier,"
    # then RFC and SVM look like suitable ensembling partners based
    # on the Pearson correlation coefficient.

    # 7a: Stacking models via logistic regression as the ensembling
    # final combiner (I think this was recommended in Corey Wade's book)
    stacking_clf = StackingClassifier(
                     estimators=[("rfc", estimator_data["rfc"]["classifier"]),
                                 ("svm", make_pipeline(StandardScaler(), estimator_data["svm"]["classifier"])),
                                 ("xgb_class", estimator_data["xgb_class"]["classifier"]),
                                 ],
                     final_estimator=LogisticRegression(),
                     # NOTE: this cv strategy should re-fit (overwrite)
                     # the previous fits I think; I did this because of the
                     # warnings related to `prefit` option in sklearn docs...
                     cv=StratifiedKFold(5),
                     stack_method="predict_proba",
                     )
    stacking_clf.fit(X_train, y_train)
    msg = "There should be three stacked estimators: XGB, RFC, SVM"
    assert len(stacking_clf.estimators_) == 3, msg
    estimator_data["stacking"] = {"classifier": stacking_clf}
    # now, ROC comparison vs. individual estimators
    fig, ax = plt.subplots()
    for estimator_name in estimator_data.keys():
        if estimator_name == "xgb_dart":
            continue
        estimator = estimator_data[estimator_name]["classifier"]
        if estimator_name == "svm":
            # need the scaler for SVM
            estimator = make_pipeline(StandardScaler(), estimator)
            estimator.fit(X_train, y_train)
        if estimator_name == "stacking":
            color = "red"
        else:
            color = "grey"
        pred_proba = estimator.predict_proba(X_test)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_proba[..., 1])
        roc_auc = metrics.auc(fpr, tpr)
        ax.plot(fpr,
                tpr,
                label=f"{estimator_name} (AUC = {roc_auc:.2f})",
                marker=".",
                alpha=0.6,
                color=color,
                lw=5)
    ax.legend()
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Stacking Classification on Test")
    ax.set_aspect("equal")
    fig.set_size_inches(8, 8)
    fig.savefig("stacking_roc.png", dpi=300)

    # 7b: Ensembling via weighted soft (and hard) voting
    # Let's use xgb_class, RFC, and SVM together again

    # Chollet recommended using a simple Nelder-Mead optimization
    # to determine weights in his ML book
    # approach also inspired by:
    # https://guillaume-martin.github.io/average-ensemble-optimization.html

    # first get the cross-validated estimates for
    # each input record for each model we want to include
    for estimator_name in estimator_data.keys():
        if estimator_name in ["xgb_dart", "stacking"]:
            continue
        estimator = estimator_data[estimator_name]["classifier"]
        if estimator_name == "svm":
            # need the scaler for SVM
            estimator = make_pipeline(StandardScaler(), estimator)

        # generate cross validated estimates for each training data point
        pred_proba = cross_val_predict(estimator=estimator,
                                       X=X_train,
                                       y=y_train,
                                       cv=StratifiedKFold(5),
                                       method="predict_proba")
        # repeat for hard voting as well:
        pred = cross_val_predict(estimator=estimator,
                                 X=X_train,
                                 y=y_train,
                                 cv=StratifiedKFold(5),
                                 method="predict")
        # the second column of the pred_proba array should
        # be the probabilities for phase separation ("1" value)
        estimator_data[estimator_name]["train_predictions_soft"] = pred_proba[..., 1]
        estimator_data[estimator_name]["train_predictions_hard"] = pred

    train_predictions_soft = np.concatenate([estimator_data["rfc"]["train_predictions_soft"][:, None],
                                             estimator_data["svm"]["train_predictions_soft"][:, None],
                                             estimator_data["xgb_class"]["train_predictions_soft"][:, None]],
                                             axis=1)
    train_predictions_hard = np.concatenate([estimator_data["rfc"]["train_predictions_hard"][:, None],
                                             estimator_data["svm"]["train_predictions_hard"][:, None],
                                             estimator_data["xgb_class"]["train_predictions_hard"][:, None]],
                                             axis=1)


    for voting_type, train_predictions in zip(["soft", "hard"],
                                              [train_predictions_soft, train_predictions_hard]):
        # next, let's try minimizing the MSE of the CV predictions
        # to obtain the weights for soft (and hard) voting
        def objective(weights):
            y_ens = np.average(train_predictions, axis=1, weights=weights)
            return metrics.mean_squared_error(y_train, y_ens)
        results_list = []
        weights_list = []
        for k in range(100):
            rng = np.random.default_rng(k)
            w0 = rng.uniform(size=train_predictions.shape[1])
            bounds = [(0,1)] * train_predictions.shape[1]
            cons = [{'type': 'eq',
                     'fun': lambda w: w.sum() - 1}]
            res = minimize(objective,
                   w0,
                   method='SLSQP', # Chollet recommended Nelder-Mead, but doesn't support constraints
                   bounds=bounds,
                   options={'disp': False, 'maxiter': 10000},
                   constraints=cons)
            results_list.append(res.fun)
            weights_list.append(res.x)
        best_score = np.min(results_list)
        best_weights = weights_list[results_list.index(best_score)]
        assert_allclose(sum(best_weights), 1.0)
        print("model order: rfc, svm, xgb_class")
        print("best_weights:", best_weights)
        fig_vote_weights, ax = plt.subplots()
        ax.bar(["RFC", "SVM", "XGB Class"],
               height=best_weights)
        ax.set_xlabel("Estimator")
        ax.set_ylabel("Weight")
        ax.set_title(f"SLSQP weights on training ({voting_type} voting)")
        fig_vote_weights.savefig(f"{voting_type}_vote_weights.png", dpi=300)


        voting_clf = VotingClassifier(estimators=[("rfc", estimator_data["rfc"]["classifier"]),
                                                  ("svm", make_pipeline(StandardScaler(), estimator_data["svm"]["classifier"])),
                                                  ("xgb_class", estimator_data["xgb_class"]["classifier"]),
                                                 ],
                                           voting=f"{voting_type}",
                                           weights=best_weights,
                                           n_jobs=-1)
        voting_clf.fit(X_train, y_train)
        msg = "There should be three stacked estimators: RFC, SVM, XGB Class"
        assert len(voting_clf.estimators_) == 3, msg
        estimator_data[f"{voting_type}_voting"] = {"classifier": voting_clf}

        # now, ROC comparison vs. individual estimators
        fig, ax = plt.subplots()
        for estimator_name in estimator_data.keys():
            if estimator_name in ["xgb_dart", "stacking"]:
                continue
            if "voting" in estimator_name and voting_type not in estimator_name:
                continue
            estimator = estimator_data[estimator_name]["classifier"]
            if estimator_name == "svm":
                # need the scaler for SVM
                estimator = make_pipeline(StandardScaler(), estimator)
                estimator.fit(X_train, y_train)
            if "voting" in estimator_name:
                color = "red"
            else:
                color = "grey"
            if voting_type == "soft":
                pred = estimator.predict_proba(X_test)[..., 1]
            else:
                pred = estimator.predict(X_test)
            fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
            roc_auc = metrics.auc(fpr, tpr)
            ax.plot(fpr,
                    tpr,
                    label=f"{estimator_name} (AUC = {roc_auc:.2f})",
                    marker=".",
                    alpha=0.6,
                    color=color,
                    lw=5)
        ax.legend()
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{voting_type} voting classification on test")
        ax.set_aspect("equal")
        fig.set_size_inches(6, 6)
        fig.savefig(f"{voting_type}_voting_roc.png", dpi=300)

    # 7c: Ensembling via entropy weighting; see section 3.2.3
    # of Kunapuli's "Ensemble Methods for Marchine Learning" (2023);
    # Let's use xgb_class, RFC, and SVM together again
    print("-" * 70)
    print("Step 7c: Start of entropy weighting ensembling")
    ent_weights = []
    ent_ensemble_members = []
    for estimator_name in estimator_data.keys():
        if estimator_name in ["xgb_dart", "stacking", "soft_voting", "hard_voting"]:
            continue
        print(f"hard prediction for {estimator_name}:", estimator_data[estimator_name]["train_predictions_hard"])
        validation_hard_preds = estimator_data[estimator_name]["train_predictions_hard"]
        ent_weights.append(1 / lib.entropy(validation_hard_preds))
        ent_ensemble_members.append(estimator_name)
    ent_weights = np.asarray(ent_weights)
    ent_weights /= np.sum(ent_weights)
    print("ent_weights:", ent_weights)
    entropy_clf = VotingClassifier(estimators=[("rfc", estimator_data["rfc"]["classifier"]),
                                              ("xgb_class", estimator_data["xgb_class"]["classifier"]),
                                              ("svm", make_pipeline(StandardScaler(), estimator_data["svm"]["classifier"])),
                                             ],
                                       voting="soft",
                                       weights=ent_weights,
                                       n_jobs=-1)

    entropy_clf.fit(X_train, y_train)
    err_msg = f"The entropy weights follow this estimator order: {ent_ensemble_members}, but the entropy VotingClassifier uses this estimator order: {entropy_clf.named_estimators.keys()}"
    assert ent_ensemble_members == list(entropy_clf.named_estimators_.keys()), err_msg
    estimator_data["entropy_weighting"] = {"classifier": entropy_clf}

    # now, ROC comparison vs. individual estimators
    fig, ax = plt.subplots()
    for estimator_name in ["rfc", "xgb_class", "svm", "entropy_weighting"]:
        estimator = estimator_data[estimator_name]["classifier"]
        if estimator_name == "svm":
            # need the scaler for SVM
            estimator = make_pipeline(StandardScaler(), estimator)
            estimator.fit(X_train, y_train)
        if "weighting" in estimator_name:
            color = "red"
        else:
            color = "grey"
        pred = estimator.predict_proba(X_test)[..., 1]
        fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
        roc_auc = metrics.auc(fpr, tpr)
        ax.plot(fpr,
                tpr,
                label=f"{estimator_name} (AUC = {roc_auc:.2f})",
                marker=".",
                alpha=0.6,
                color=color,
                lw=5)
    ax.legend()
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Entropy weighted voting classification on test")
    ax.set_aspect("equal")
    fig.set_size_inches(6, 6)
    fig.savefig("entopy_weighted_voting_roc.png", dpi=300)
    print("-" * 70)

    # 7d: Ensembling via Dempster-Shafer combination; see section 3.2.4
    # of Kunapuli's "Ensemble Methods for Marchine Learning" (2023);
    # Let's use xgb_class, RFC, and SVM together again
    print("-" * 70)
    print("Step 7d: Start of Dempster-Shafer ensembling")
    # ROC comparison vs. individual estimators
    fig, ax = plt.subplots()
    for estimator_name in ["rfc", "xgb_class", "svm", "DST"]:
        color = "grey"
        if estimator_name != "DST":
            estimator = estimator_data[estimator_name]["classifier"]
            if estimator_name == "svm":
                # need the scaler for SVM
                estimator = make_pipeline(StandardScaler(), estimator)
                estimator.fit(X_train, y_train)
            pred = estimator.predict_proba(X_test)[..., 1]
        else:
            color = "red"
            # TODO: we're using the normalized DST beliefs here...
            # is that allowed for ROC AUC?
            pred = lib.dempster_shafer_pred([estimator_data["rfc"]["classifier"],
                                             estimator_data["svm"]["classifier"],
                                             estimator_data["xgb_class"]["classifier"]],
                                            X_train,
                                            y_train,
                                            X_test)[1]
        fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
        roc_auc = metrics.auc(fpr, tpr)
        ax.plot(fpr,
                tpr,
                label=f"{estimator_name} (AUC = {roc_auc:.2f})",
                marker=".",
                alpha=0.6,
                color=color,
                lw=5)
    ax.legend()
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("DST combination classification on test\n(TODO: DST normalized belief allowed?)")
    ax.set_aspect("equal")
    fig.set_size_inches(6, 6)
    fig.savefig("DST_combination_roc.png", dpi=300)
    print("-" * 70)

    # Step 7e: Prepare Cesar's CG/AA-MD simulation data(frame)
    # for feature importance analysis by using our best classifier
    # (SVM) to predict the phase separated status of each of his
    # records
    print("-" * 70)
    print("Step 7e: Use experiment-based SVM model to add labels"
          " to Cesar's CG/AA-MD data, as a precursor to feature importance\n"
          "analysis of the PEO/DEX system.")
    # So far, on the binary phase separation PEO/DEX experimental data
    # from Mihee, which only has % PEO, % DEX, and phase separation status (yes/no),
    # SVM has been the most accurate ML model. So, let's try using that model
    # to estimate the phase separated labels on Cesar's CG/AA-MD sim data
    experiment_svm_clf = estimator_data["svm"]["classifier"]
    experiment_svm_clf.fit(X_train, y_train)
    y_pred = experiment_svm_clf.predict(df_cesar_combined[["WT% DEX", "WT% PEO"]].to_numpy())
    assert y_pred.shape[0] == df_cesar_combined.shape[0]
    # plot the phase map with predicted labels, for ease
    # of side-by-side comparison with the original expt
    # data
    lib.plot_input_data_cesar_MD(df=df_cesar_combined, y_pred=y_pred)
    # the phase separation labels in the plot look sensible, so assign them
    y_pred_cesar_md = y_pred # noqa
    print("-" * 70)

    # Step 8: Feature Importance Analysis
    print("-" * 70)
    print("Step 8: Feature Importance Analysis of Mihee Expt Data")
    for key, subdict in estimator_data.items():
        if key == "hard_voting":
            # can't calculate SHAP without probabilities
            continue
        classifier = subdict["classifier"]
        # some of the classifiers stored in the dict may not
        # be fit...
        classifier.fit(X_train, y_train)
        # check that the classifier was fit reasonably
        # above (the workflow is getting large and hard to manage...)
        y_preds = classifier.predict(X_test)
        balanced_acc = metrics.balanced_accuracy_score(y_test, y_preds)
        msg = f"estimator {key} has {balanced_acc = }; may not have been fit?"
        assert balanced_acc > 0.66, msg
        try:
            explainer = shap.Explainer(classifier)
        except TypeError:
            # handle the non-tree/general cases
            explainer = shap.KernelExplainer(classifier.predict_proba,
                                             X_train)
        shap_values = explainer.shap_values(X_train)
        positive_class_shap_values = lib.get_positive_shap_values(shap_values)
        assert positive_class_shap_values.shape == X_train.shape
        print("Successfully calculated SHAP values for "
              f"(classifier: {key}) on training data")
        lib.plot_ma_shap_vals_per_model(shap_values=positive_class_shap_values,
                                        feature_names=df.columns[1:-1],
                                        fig_title=f"{key} model",
                                        fig_name=f"{key}_SHAP_mean_absolute.png")
    print("-" * 70)

    # Step 8b: Feature Importance Analysis of CG/AA-MD data
    # TODO: reduce code duplication on SHAP/feat imp analyses?
    print("-" * 70)
    print("Step 8b: Feature Importance Analysis of CG/AA-MD Data")
    # perform SHAP analysis on random forest and SVM
    # TODO: there's no OOB score for SVM, so should eventually check
    # on validation...
    rf = RandomForestClassifier(random_state=0,
                                oob_score=metrics.balanced_accuracy_score)
    rf.fit(df_cesar_combined.to_numpy(), y_pred_cesar_md)
    oob_bal_acc_score = rf.oob_score_
    explainer = shap.Explainer(rf)
    shap_values = explainer.shap_values(df_cesar_combined.to_numpy())
    positive_class_shap_values_rfc = lib.get_positive_shap_values(shap_values)
    lib.plot_ma_shap_vals_per_model(shap_values=positive_class_shap_values_rfc,
                                    feature_names=df_cesar_combined.columns,
                                    fig_title=f"Random Forest model\n(oob balanced accuracy = {oob_bal_acc_score:.3f})",
                                    fig_name="RF_SHAP_mean_absolute_MD.png",
                                    top_feat_count=10)

    # perform an EBM analysis with feature interactions turned off
    # (because of: https://github.com/interpretml/interpret/issues/513)
    # also playing with some overfit guards...
    ebm = ExplainableBoostingClassifier(interactions=0,
                                        early_stopping_tolerance=0.0001,
                                        early_stopping_rounds=25)
    ebm.fit(df_cesar_combined.to_numpy(), y_pred_cesar_md)
    ebm_pred = ebm.predict(df_cesar_combined.to_numpy())
    ebm_bal_acc = metrics.balanced_accuracy_score(y_pred_cesar_md, ebm_pred)
    # around 0.85 (so maybe not quite as overfit as below?)
    explain_data = ebm.explain_global().data()
    ebm_feature_scores = np.asarray(explain_data["scores"]) # shape (926,)

    svm = SVC(gamma="auto", probability=True)
    svm = make_pipeline(StandardScaler(), svm)
    svm.fit(df_cesar_combined.to_numpy(), y_pred_cesar_md)
    # TODO: need actual validation for SVM, not acc on training itself...
    svm_pred = svm.predict(df_cesar_combined.to_numpy())
    svm_bal_acc = metrics.balanced_accuracy_score(y_pred_cesar_md, svm_pred)
    # cache SVM SHAP because it takes several minutes
    # to compute
    svm_shap_positive_cache_file = "svm_shap_cache_positive_vals.npy"
    if not Path(svm_shap_positive_cache_file).exists():
        explainer = shap.KernelExplainer(svm.predict_proba,
                                         df_cesar_combined.to_numpy())
        shap_values = explainer.shap_values(df_cesar_combined.to_numpy())
        positive_class_shap_values_svm = lib.get_positive_shap_values(shap_values)
        with open(svm_shap_positive_cache_file, 'wb') as f:
            np.save(f, positive_class_shap_values_svm)
    else:
        with open(svm_shap_positive_cache_file, 'rb') as f:
            positive_class_shap_values_svm = np.load(f)
    lib.plot_ma_shap_vals_per_model(shap_values=positive_class_shap_values_svm,
                                    feature_names=df_cesar_combined.columns,
                                    fig_title=f"SVM model\n(training balanced accuracy = {svm_bal_acc:.3f})",
                                    fig_name="SVM_SHAP_mean_absolute_MD.png",
                                    top_feat_count=10)

    # might as well include the "native" RF feature importances into the mix
    native_rf_feature_scores = rf.feature_importances_

    # Try using LIME with RF for feature importances:
    rf_lime_scores = lib.build_lime_data(X=df_cesar_combined, model=rf)
    assert rf_lime_scores.shape == df_cesar_combined.shape


    # add XGBoost + SHAP feature importances into the mix
    xgb_cls = xgb.XGBClassifier(random_state=0)
    xgb_cls.fit(df_cesar_combined.to_numpy(), y_pred_cesar_md)
    explainer = shap.Explainer(xgb_cls)
    shap_values_xgb_cls = explainer.shap_values(df_cesar_combined.to_numpy())
    positive_class_shap_values_xgb_cls = lib.get_positive_shap_values(shap_values_xgb_cls)
    # TODO: validation on the classifier for xgb above...

    # Try using LIME with XGB for feature importances:
    xgb_cls_lime_scores = lib.build_lime_data(X=df_cesar_combined, model=xgb_cls)
    assert xgb_cls_lime_scores.shape == df_cesar_combined.shape


    # add lightgbm + SHAP feature importances into the mix
    lgb_bst = lgb.LGBMClassifier(n_estimators=500,
                                 objective="binary",
                                 n_jobs=-1,
                                 importance_type="split",
                                 random_state=0)
    lgb_bst.fit(df_cesar_combined.to_numpy(), y_pred_cesar_md)
    explainer = shap.Explainer(lgb_bst)
    shap_values_lgb = explainer.shap_values(df_cesar_combined.to_numpy())
    positive_class_shap_values_lgb = lib.get_positive_shap_values(shap_values_lgb)
    # TODO: validation on the classifier for lightgbm above...

    # Add SelectKBest + mutual_info_classif feature importance analysis into the mix
    # let's require a selection of the top 10 features since
    # that's our current top feature count for consideration in the
    # consensus analysis below, but I'm not sure it matters since
    # we can get all feature scores out of the analysis
    k_best_metrics = [mutual_info_classif, f_classif]
    k_best_scores = lib.select_k_best_scores(X=df_cesar_combined,
                                             y=y_pred_cesar_md,
                                             k=10,
                                             metrics=k_best_metrics)

    # might as well add lightgbm "native" feature importances into
    # the consensus feature importance analysis mix
    lgb_native_feature_importances = lgb_bst.feature_importances_

    # try using ExtraTrees in the consensus feature importance
    # analysis as well
    extra_t_cls = ExtraTreesClassifier(random_state=0,
                                       n_estimators=500,
                                       bootstrap=True,
                                       oob_score=metrics.balanced_accuracy_score)
    extra_t_cls.fit(df_cesar_combined.to_numpy(), y_pred_cesar_md)
    oob_bal_acc_score = extra_t_cls.oob_score_
    expected_et_oob = 0.70
    msg = f"{oob_bal_acc_score = } for extra trees classifier, but expected at least {expected_et_oob}"
    assert oob_bal_acc_score >= expected_et_oob, msg
    extra_t_native_feat_imp = extra_t_cls.feature_importances_

    # Try using LIME with ExtraTrees for feature importances:
    extra_t_cls_lime_scores = lib.build_lime_data(X=df_cesar_combined, model=extra_t_cls)
    assert extra_t_cls_lime_scores.shape == df_cesar_combined.shape


    # try to find consensus amongst the important
    # features from different ML models
    (ranked_feature_names,
     ranked_feature_counts,
     num_input_models) = lib.feature_importance_consensus(
                                     pos_class_feat_imps=[positive_class_shap_values_rfc,
                                                          positive_class_shap_values_svm,
                                                          ebm_feature_scores,
                                                          native_rf_feature_scores,
                                                          positive_class_shap_values_xgb_cls,
                                                          positive_class_shap_values_lgb,
                                                          lgb_native_feature_importances,
                                                          rf_lime_scores,
                                                          xgb_cls_lime_scores,
                                                          extra_t_cls_lime_scores,
                                                          extra_t_native_feat_imp] +
                                                          k_best_scores,
                                     feature_names=df_cesar_combined.columns,
                                     top_feat_count=10)
    lib.plot_feat_import_consensus(ranked_feature_names=ranked_feature_names,
                                   ranked_feature_counts=ranked_feature_counts,
                                   num_input_models=num_input_models,
                                   top_feat_count=10)
    # perform EBM analysis
    # TODO: no OOB score available as far as I know, so should eventually
    # check on validation...
    # TODO: are feature interactions really believable/useful here?
    # not enough records... and it adds a lot of calculation time...
    # see: https://github.com/interpretml/interpret/issues/513

    # cache the interactions-enabled EBM work because it
    # is fairly slow
    cached_ebm_interact_explain_data = "cached_ebm_interact_explain_data.p"
    if not Path(cached_ebm_interact_explain_data).exists():
        ebm = ExplainableBoostingClassifier()
        ebm.fit(df_cesar_combined.to_numpy(), y_pred_cesar_md)
        ebm_pred = ebm.predict(df_cesar_combined.to_numpy())
        ebm_bal_acc = metrics.balanced_accuracy_score(y_pred_cesar_md, ebm_pred)
        explain_data = ebm.explain_global().data()
        with open(cached_ebm_interact_explain_data, "wb") as cache_file:
            pickle.dump(explain_data, cache_file)
    else:
        with open(cached_ebm_interact_explain_data, "rb") as cache_file:
            explain_data = pickle.load(cache_file)
    lib.plot_ebm_data(explain_data=explain_data,
                      original_feat_names=df_cesar_combined.columns,
                      fig_title=f"Top 10 EBM features\n(training balanced accuracy = {ebm_bal_acc})",
                      fig_name="EBM_top_10_features_cesar_md.png",
                      top_feat_count=10)
    print("-" * 70)



if __name__ == "__main__":
    main()
