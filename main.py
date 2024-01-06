import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import minimize
import pandas as pd
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, GridSearchCV,
                                     cross_val_predict)
from sklearn.ensemble import (RandomForestClassifier, StackingClassifier,
                              VotingClassifier)
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt


import lib


def main():
    # Step 1: Read in the experimental data/format it appropriately
    df = pd.read_excel("data/mihee_peo_dextran_phase_map_experimental.xlsx")
    X, y = lib.preprocess_data(df=df)
    lib.plot_input_data(X, y)

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



if __name__ == "__main__":
    main()
