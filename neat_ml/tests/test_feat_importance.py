import os

from neat_ml import lib

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import shap


@pytest.mark.parametrize("n_features, important_index", [
    (3, 0),
    (5, 2),
    ])
def test_individual_shap_absolute_plots(tmp_path, n_features, important_index):
    plot_filename = "synthetic_SHAP_mean_absolute.png"
    expected_plot_file = tmp_path / plot_filename
    X, y = make_classification(n_samples=500,
                               n_features=n_features,
                               n_clusters_per_class=1,
                               n_informative=1,
                               n_redundant=0,
                               random_state=0,
                               shuffle=False)
    X_tmp = X.copy()
    # carefully tune the location of the important
    # feature for testing purposes
    X[:, [0, important_index]] = X_tmp[:, [important_index, 0]]
    clf = RandomForestClassifier(random_state=0)
    clf.fit(X, y)
    explainer = shap.Explainer(clf)
    # shap_values(X) is an ndarray with shape (500, n_features, 2) with shap versions >= 0.45.0
    # and is a list in versions prior to that
    # see: https://github.com/shap/shap/pull/3318
    shap_vals = explainer.shap_values(X)
    positive_class_shap_values = lib.get_positive_shap_values(shap_vals)

    cwd = os.getcwd()
    os.chdir(tmp_path)
    actual_fig = lib.plot_ma_shap_vals_per_model(shap_values=positive_class_shap_values,
                                                 feature_names=[str(x) for x in range(n_features)],
                                                 fig_title="synthetic SHAP test",
                                                 fig_name=plot_filename)
    os.chdir(cwd)
    # first check that plot was produced
    assert expected_plot_file.is_file()
    # now check that the first horizontal bar
    # has the largest width (most important feature)
    axis = actual_fig.get_axes()[0]
    actual_patches = axis.patches
    actual_bar_widths = []
    for patch in actual_patches:
        actual_bar_widths.append(patch.get_width())
    assert len(actual_bar_widths) == n_features
    assert np.argmax(actual_bar_widths) == important_index
