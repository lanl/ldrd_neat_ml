import os
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import ternary
import pandas as pd


# these features were considered potentially interesting
# for machine learning-based prediction of polymer
# properties by Mihee, Cesar, and Tyler during a discussion
# on Nov. 3/2023
features_of_interest = ["volume_fraction",
                        # temperature at which light transmission of
                        # polymer mixture changes (within biological range):
                        "temp_light_transmission", 
                        # polydispersity index (dispersity) -- different
                        # polymer lengths that result from imperfections in
                        # human synthesis/manufacturing processes:
                        "PDI",
                        # weight-average molar mass:
                        "M_w",
                        "backbone_or_torsion_angles",
                        "radius_of_gyration",
                        # these maps might be intra or even inter-molecular:
                        "contact_or_adjacency_maps",
                        # H-bond contacts in first solvation shell?
                        "h_bond_contacts",
                        # Mihee can measure interfacial tension
                        "interfacial_tension",
                        # Viscoscity may matter for cytometry, but not clear
                        # if it would matter for polymer phase separatation/
                        # microparticle formation:
                        "viscosity"]

def preprocess_data(df):
    # Accepts an input DataFrame and produces the typical
    # training (X) and prediction (y) values needed for ML
    X = df[["Dextran (wt%)",  "PEO (wt%)"]].to_numpy()
    y = df["Phase Separated"].to_numpy()
    y[y == "Yes"] = 1
    y[y == "No"] = 0
    y = y.astype(np.int32)
    return X, y


def plot_input_data_cesar_CG(df, y_pred=None):
    # Produce a simple scatter plot of Cesar's CG
    # MD input data, meant for side-by-side comparison
    # with the expt PEO/DEX binary phase separation data from
    # Mihee
    fig, ax = plt.subplots(1, 1)
    if y_pred is not None:
        c = y_pred
        title_addition = "(phase sep labels from SVM)"
        fig_name_addition = "predicted"
    else:
        c = "gray"
        title_addition = "(phase sep labels unknown)"
        fig_name_addition = "original"
    im = ax.scatter(df["WT% DEX"], df["WT% PEO"], c=c)
    ax.set_aspect("equal")
    ax.set_xlabel("Dextran (wt %)")
    ax.set_ylabel("PEO (wt %)")
    ax.set_title("Cesar CG-MD input data\n"
                 f"{title_addition}")
    if y_pred is not None:
        fig.colorbar(im, ax=ax, shrink=0.9)
    fig.savefig(f"cesar_cg_md_input_data_{fig_name_addition}.png",
                dpi=300)


def plot_input_data(X, y):
    # plot scatter + heatmap diagrams of the input data
    # from Mihee (currently just binary PEO/dextran data,
    # but will eventually include block copolymer to represent
    # ternary mixture data)
    fig, axes = plt.subplots(1, 2)

    # simple/raw scatter plot of the raw data
    # from Mihee:
    ax_scatter = axes[0]
    ax_scatter.set_title("Scatter plot of binary phase separation data\n"
                         "(1.0 is phase separated)")
    im = ax_scatter.scatter(X[..., 0],
                            X[..., 1],
                            c=y)

    # a somewhat-prettier representation of the data
    # with some interpolation, to get us thinking about
    # "gaps"/areas to fill in for the data (and what to do
    # in the ternary case/how to represent that)
    num_unique_dex_vals = np.unique(X[..., 0]).size
    num_unique_peo_vals = np.unique(X[..., 1]).size
    # let's make a grid for all possible combos of
    # the binary mixture, with np.nan a sensible fill
    # value since we'll have some missing points that
    # i.e., Mihee didn't collect (at the time of writing 34/48
    # grid points were sampled)
    grid = np.full(shape=(num_unique_dex_vals, num_unique_peo_vals),
                   fill_value=np.nan,
                   dtype=np.float64)
    ax_im = axes[1]
    ax_im.set_title("Interpolated heatmap of binary phase\nseparation data")
    # use a trick with rankdata to assign
    # grid positions
    x_grid_positions = rankdata(X[..., 0], method="dense") - 1
    y_grid_positions = rankdata(X[..., 1], method="dense") - 1
    # assign value to grid positions based on the ordering
    for row, column, value in zip(x_grid_positions,
                                  y_grid_positions,
                                  y):
        grid[row, column] = value
    # NOTE: the "interpolation" isn't really doing much
    # here, but the imshow visualization is still helpful
    # to show the gaps in collected experimental data
    ax_im.imshow(grid.T,
                 origin="lower",
                 extent=[0, 13, 0, 13],
                 interpolation="nearest")
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("Dextran (wt %)")
        ax.set_ylabel("PEO (wt %)")
    fig.colorbar(im, ax=ax_scatter, shrink=0.3)
    fig.set_size_inches(9, 9)
    fig.savefig("binary_phase_sep_heat_map.png", dpi=300)


hyper_param_dict = {"rfc": {"max_depth": [1, 10, 100, None],
                            "min_samples_split": [2, 4, 10],
                            },
                    "xgb_class": {"n_estimators": [20, 100, 2_000],
                                  "subsample": [1, 0.9, 0.8],
                                  "colsample_bytree": [1, 0.9, 0.7]},
                    "xgb_dart": {"n_estimators": [20, 100, 300],
                                  "subsample": [1, 0.8],
                                  "colsample_bytree": [1, 0.9, 0.7]},
                    "svm": {"C": [1, 10],
                            "kernel": ["linear", "rbf"]},
                    }

def color_df(styler):
    # TODO: set vmin/vmax based on actual
    # range of values in DataFrame
    vmin = 0.8
    vmax = 1.0
    styler.background_gradient(axis=None, vmin=vmin, vmax=vmax, cmap="viridis")
    return styler


def entropy(y):
    # see section 3.2.3 of Kunapuli's "Ensemble Methods for Machine
    # Learning" (2023)
    counts = np.unique(y, return_counts=True)[1]
    p = np.array(counts.astype(np.float64)) / len(y)
    ent = -p.T @ np.log2(p)
    return ent


def dempster_shafer_pred(estimators,
                         X_train,
                         y_train,
                         X_test):
    # train a list of estimators, and then use
    # Dempster-Shafer theory to combine their
    # predictions on test data
    # see section 3.2.4 of Kunapuli's
    # "Ensemble Methods for Marchine Learning" (2023)
    trained_estimators = []
    for estimator in estimators:
        # probably safe to scale the data for each estimator?
        estimator = make_pipeline(StandardScaler(), estimator)
        trained_estimators.append(estimator.fit(X_train, y_train))
    test_preds = np.empty(shape=(len(estimators), X_test.shape[0]))
    for index, estimator in enumerate(trained_estimators):
        test_preds[index, ...] = estimator.predict_proba(X_test)[..., 1]
    # column 0 is NOT phase sep
    bpa_0 = 1 - np.prod(test_preds, axis=0)
    # column 1 IS phase sep
    bpa_1 = 1 - np.prod(1 - test_preds, axis=0)
    belief_0 = bpa_0 / (1 - bpa_0)
    belief_1 = bpa_1 / (1 - bpa_1)
    belief = np.vstack([belief_0, belief_1]).T
    # y_pred_ds now contains the hard 0/1 class predictions
    # based on the highest "belief" score
    y_pred_ds = np.argmax(belief, axis=1)
    # for ROC curve we don't want hard decision though, let's
    # try using normalized beliefs?
    # TODO: check literature to see if this is acceptable?
    Z = belief_0 + belief_1 + 1
    y_normalized_beliefs = belief_1 / Z
    return y_pred_ds, y_normalized_beliefs


def plot_tri_phase_diagram(X,
                           y,
                           plot_path,
                           plot_name="ternary.png",
                           bottom_label_z="",
                           right_label_y="",
                           left_label_x="",
                           clockwise=True):
    if X.shape[1] != 3:
        raise ValueError("Ternary plot requires input with three variables.")
    if np.unique(np.sum(X, axis=1)).size != 1:
        raise ValueError("The ternary phase diagram inputs do not sum to a constant value.")
    figure, tax = ternary.figure(scale=100)
    tax.clear_matplotlib_ticks()
    tax.boundary(linewidth=2.0)
    tax.gridlines(color="blue", multiple=5)
    # need to swap columns to match convention
    # of:
    # https://en.wikipedia.org/wiki/Ternary_plot#Example
    X_loc = X.copy()
    # TODO: clockwise vs. counterclockwise conventions handled
    # more gracefully? this is pretty confusing manipulation!
    offset = 0.15
    if clockwise:
        X_loc[..., [0, 1]] = X_loc[..., [1, 0]]
        tax.right_axis_label(f"{right_label_y}", offset=offset)
        tax.bottom_axis_label(f"{bottom_label_z}", offset=offset)
        tax.left_axis_label(f"{left_label_x}", offset=offset)
    else:
        X_loc[..., [0, 2]] = X_loc[..., [2, 0]]
        X_loc[..., [0, 1]] = X_loc[..., [1, 0]]
        tax.right_axis_label(f"{left_label_x}", offset=offset)
        tax.bottom_axis_label(f"{right_label_y}", offset=offset)
        tax.left_axis_label(f"{bottom_label_z}", offset=offset)
    tax.scatter(X_loc, c=y / y.max())
    tax.set_title("Ternary Phase Diagram (synthetic data for now)\n", fontsize=10)
    tax.get_axes().axis('off')
    tax.ticks(axis='lbr',
              multiple=10,
              linewidth=1,
              offset=0.025,
              clockwise=clockwise)
    # this is apparently needed on some platforms for
    # axis labels to show up; see:
    # https://github.com/marcharper/python-ternary/blob/master/README.md?plain=1#L472
    tax._redraw_labels()
    figure.savefig(os.path.join(plot_path, f"{plot_name}"), dpi=300)
    return figure


def plot_ma_shap_vals_per_model(shap_values,
                                feature_names,
                                fig_title: str,
                                fig_name: str,
                                top_feat_count: Optional[int] = None):
    # plot the mean absolute SHAP values for
    # any models
    # NOTE: shap_values should be for the "positive" class,
    # though for now it probably doesn't matter since we have
    # a binary classification with symmetric feature importances
    fig, ax = plt.subplots(1, 1)
    abs_shap_values = np.absolute(shap_values)
    mean_abs_shap_values = np.mean(abs_shap_values, axis=0)
    if top_feat_count is None:
        y_pos = np.arange(len(feature_names))
    else:
        # useful to pick only top few features when there are tons
        sort_idx = np.argsort(mean_abs_shap_values)[::-1]
        feature_names = feature_names[sort_idx][:top_feat_count]
        mean_abs_shap_values = mean_abs_shap_values[sort_idx][:top_feat_count]
        y_pos = np.arange(top_feat_count)
    ax.barh(y_pos, mean_abs_shap_values)
    ax.set_title(fig_title, fontsize=6)
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_yticks(y_pos, labels=feature_names)
    fig.set_size_inches(3, 3)
    fig.tight_layout()
    fig.savefig(f"{fig_name}", dpi=300)
    return fig


def read_in_cesar_cg_md_data():
    # CG-MD gyration data:
    df_cesar_cg_gyr_persistence = pd.read_excel("neat_ml/data/CG-PHASE-DESCRIPTORS.xlsx",
                                                sheet_name=0)
    df_cesar_cg_gyr_persistence.dropna(how='all', inplace=True) # shape (49, 10)
    assert df_cesar_cg_gyr_persistence.isna().sum().sum() == 0
    # CG-MD RDF data:
    df_cesar_cg_rdf = pd.read_excel("neat_ml/data/CG-PHASE-DESCRIPTORS.xlsx",
                                    sheet_name=1)
    df_cesar_cg_rdf.dropna(how='all', inplace=True) # shape (49, 904)
    assert df_cesar_cg_rdf.isna().sum().sum() == 0
    # fuse Cesar's CG-MD data on the WT % columns
    df_cesar_cg = df_cesar_cg_gyr_persistence.merge(df_cesar_cg_rdf,
                                                    on=["WT% DEX",
                                                        "WT% PEO",
                                                        "WT% WATER"])
    # we've joined on three columns, so check that the shape/properties
    # match expectations
    assert df_cesar_cg.isna().sum().sum() == 0
    expected_rows = df_cesar_cg_rdf.shape[0]
    expected_cols = (df_cesar_cg_rdf.shape[1] +
                     df_cesar_cg_gyr_persistence.shape[1] -
                     3)
    assert df_cesar_cg.shape == (expected_rows, expected_cols)
    # Cesar's binary CG MD simulation data has a different number
    # of records, and different set of PEO/dex percentages, than
    # Mihee's original binary experimental data
    # Mihee's data has shape (34, 4) and Cesar's (49, 10) for gyration
    # and (49, 904) for RDF --> (49, 911) combined
    return df_cesar_cg
