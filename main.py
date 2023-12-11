import numpy as np
import pandas as pd
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, GridSearchCV,
                                     cross_val_predict)
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
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
    fig.savefig(f"stacking_roc.png", dpi=300)




if __name__ == "__main__":
    main()
