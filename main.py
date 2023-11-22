import numpy as np
import pandas as pd
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, GridSearchCV,
                                     cross_val_predict)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import lib


def main():
    # Step 1: Read in the experimental data/format it appropriately
    df = pd.read_excel("data/mihee_peo_dextran_phase_map_experimental.xlsx")
    X, y = lib.preprocess_data(df=df)

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



if __name__ == "__main__":
    main()
