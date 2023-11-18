import numpy as np
import pandas as pd
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold)
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
    # TODO: add hyperparameter optimization

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
                      "svm": {"classifier": make_pipeline(StandardScaler(), SVC(gamma='auto'))},
                      }

    for estimator_name in estimator_data.keys():
        scores = cross_val_score(estimator_data[estimator_name]["classifier"],
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
    # TODO


if __name__ == "__main__":
    main()
