import importlib.resources
from collections import defaultdict
import re
import os
from typing import Optional, Sequence, Tuple
import numpy as np
import numpy.typing as npt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import lime
import joblib
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
import ternary
import pandas as pd
from PIL import Image
import skimage
from tqdm import tqdm

memory = joblib.Memory("joblib_cache", verbose=0)

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


def plot_input_data_cesar_MD(df,
                             title="Cesar MD input data\n",
                             fig_name="cesar_md_input_data_",
                             title_addition=None,
                             y_pred=None):
    # Produce a simple scatter plot of Cesar's
    # MD input data, meant for side-by-side comparison
    # with the expt PEO/DEX binary phase separation data from
    # Mihee
    fig, ax = plt.subplots(1, 1)
    if y_pred is not None:
        c = y_pred
        if title_addition is None:
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
    ax.set_title(f"{title}"
                 f"{title_addition}")
    if y_pred is not None:
        fig.colorbar(im, ax=ax, shrink=0.9)
    fig.savefig(f"{fig_name}{fig_name_addition}.png",
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
    data = importlib.resources.files("neat_ml").joinpath("data/CG-PHASE-DESCRIPTORS.xlsx")
    df_cesar_cg_gyr_persistence = pd.read_excel(data,
                                                sheet_name=0)
    df_cesar_cg_gyr_persistence.dropna(how='all', inplace=True) # shape (49, 10)
    assert df_cesar_cg_gyr_persistence.isna().sum().sum() == 0
    # CG-MD RDF data:
    df_cesar_cg_rdf = pd.read_excel(data,
                                    sheet_name=1)
    # use sensible column names for RDF values,
    # otherwise we end up with unlabelled floats
    df_cesar_cg_rdf = _add_df_col_prefix(df=df_cesar_cg_rdf,
                                         start_index=3,
                                         prefix="RDF_")
    df_cesar_cg_rdf.dropna(how='all', inplace=True) # shape (49, 904)
    assert df_cesar_cg_rdf.isna().sum().sum() == 0
    # fuse Cesar's CG-MD data on the WT % columns
    df_cesar_cg = _merge_dfs(df_cesar_cg_gyr_persistence, df_cesar_cg_rdf)
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

    # to clearly distinguish CG from AA MD data, let's prefix
    # the columns appropriately
    # careful of mutability here:
    # https://github.com/pandas-dev/pandas/issues/34364#issuecomment-1960548120
    df_cesar_cg = _add_df_col_prefix(df=df_cesar_cg,
                                     start_index=3,
                                     prefix="CG_")
    return df_cesar_cg


def plot_ebm_data(explain_data: dict,
                  original_feat_names,
                  fig_title: str,
                  fig_name: str,
                  top_feat_count: int = 10):
    # plot top top_feat_count features from
    # ExplainableBoostingClassifier data
    fig, ax = plt.subplots(1, 1)
    ax.set_title(f"{fig_title}", fontsize=6)
    y_pos = np.arange(top_feat_count)
    feature_scores = np.asarray(explain_data["scores"])
    feature_names = np.asarray(explain_data["names"])
    rank_indices = np.argsort(feature_scores)[::-1]
    top_feature_scores = feature_scores[rank_indices][:top_feat_count]
    top_feature_names = feature_names[rank_indices][:top_feat_count]
    # TODO: probably need more robust mapping of
    # the feature names from EBM machinery back to
    # original names; this is a quick hack...
    remapped_top_feature_names = []
    for feature_name in top_feature_names:
        feature_1_name = feature_name.split("&")[0].strip()
        feature_2_name = feature_name.split("&")[1].strip()
        feature_1_index = int(re.sub(r"\D", "", feature_1_name))
        feature_2_index = int(re.sub(r"\D", "", feature_2_name))
        feature_1_name = original_feat_names[feature_1_index]
        feature_2_name = original_feat_names[feature_2_index]
        remapped_top_feature_names.append(f"{feature_1_name} & {feature_2_name}")
    ax.barh(y_pos, top_feature_scores)
    ax.set_yticks(y_pos, labels=remapped_top_feature_names)
    ax.set_xlabel("mean abs score")
    fig.set_size_inches(6, 3)
    fig.tight_layout()
    fig.savefig(f"{fig_name}", dpi=300)


def retrieve_image_dims(image_path: str) -> tuple[int, int]:
    with Image.open(image_path) as img:
        width, height = img.size
        return width, height


def check_image_dim_consistency(list_img_filepaths: Sequence[str]) -> None:
    # check that a list of image filepaths
    # all have the same image pixel dims
    reference_dims = retrieve_image_dims(list_img_filepaths[0])
    for img_path in list_img_filepaths[1:]:
        actual_dims = retrieve_image_dims(img_path)
        if actual_dims != reference_dims:
            msg = f"Image {reference_dims = } but {actual_dims =} for {img_path}"
            raise ValueError(msg)


def build_df_from_exp_img_paths(list_img_filepaths: Sequence[str]) -> pd.DataFrame:
    # take the % PEO / % DEX platereader image filepaths
    # and construct the initial skeleton of a useful DataFrame
    prog = re.compile(r".*DEX(\d+)wt_,PEO(\d+)wt_\.tiff")
    data_dict: dict = {"WT% PEO": [],
                       "WT% DEX": []}
    data_dict["image_filepath"] = list_img_filepaths
    for img_path in list_img_filepaths:
        match = prog.search(img_path)
        if match is not None:
            dex_percent = float(match.group(1))
            peo_percent = float(match.group(2))
            data_dict["WT% PEO"].append(peo_percent)
            data_dict["WT% DEX"].append(dex_percent)
    df = pd.DataFrame.from_dict(data_dict)
    return df


def skimage_hough_transform(df: pd.DataFrame,
                            debug: bool = False) -> None:
    # given the DataFrame of plate reader data/image
    # filepaths, use sklearn Hough transforms to estimate
    # the average diameters of the bubbles in each image
    median_droplet_radii = np.empty(shape=(df.shape[0]),
                                    dtype=np.float64)
    # 2 % PEO/ 2 % DEX as "background:"
    background_filepath = (df.loc[(df["WT% PEO"] == 2) & (df["WT% DEX"] == 2)]).image_filepath.values[0]
    background = skimage.io.imread(background_filepath)
    background = skimage.util.img_as_ubyte(background)
    # threshold for background determined empirically
    background_threshold = np.median(background) + 15
    for index, row in tqdm(df.iterrows(),
                           total=df.shape[0],
                           desc="skimage_hough_transform"):
        img_filepath = row.image_filepath
        image = skimage.io.imread(img_filepath) # shape: (2052, 2456)
        image = skimage.util.img_as_ubyte(image)
        # anything darker than the background threshold
        # should be set back to the median; helps remove
        # the background "dots"
        image[image < background_threshold] = np.median(image)
        edges = skimage.feature.canny(image,
                                      sigma=3,
                                      low_threshold=10,
                                      high_threshold=50)
        hough_radii = np.arange(2, 22, 2)
        hough_res = skimage.transform.hough_circle(edges, hough_radii)
        accums, cx, cy, radii = skimage.transform.hough_circle_peaks(hough_res,
                                                                     hough_radii,
                                                                     total_num_peaks=12)
        if radii.size == 0:
            median_droplet_radius = 0
        else:
            median_droplet_radius = np.median(radii)
        median_droplet_radii[index] = median_droplet_radius
        wt_dex = row["WT% DEX"]
        wt_peo = row["WT% PEO"]
        if debug:
            # sample/debug plots
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            image = skimage.color.gray2rgb(image)
            for center_y, center_x, radius in zip(cy, cx, radii):
                circy, circx = skimage.draw.circle_perimeter(center_y, center_x, radius,
                                    shape=image.shape)
                image[circy, circx] = (220, 20, 20)
                ax.imshow(image, cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"Median droplot radius: {median_droplet_radius}")
            fig.savefig(f"hough_transform_index_{index}_{wt_peo}_peo_{wt_dex}_dex.png", dpi=300)
            matplotlib.pyplot.close()
    df["median_radii_skimage_hough"] = median_droplet_radii


def read_in_cesar_all_atom_md_data():
    data = importlib.resources.files("neat_ml").joinpath("data/AA-PHASE-DESCRIPTORS.xlsx")
    # all-atom enthalpy data:
    df_cesar_aa_enthalpy = pd.read_excel(data,
                                         sheet_name=0)
    # some empty (NaN) rows and columns to filter out:
    for axis in [0, 1]:
        df_cesar_aa_enthalpy.dropna(axis=axis, how='all', inplace=True)
    # make sure no NaNs survived the filtering of
    # AA enthalpy data:
    assert df_cesar_aa_enthalpy.isna().sum().sum() == 0 # shape: (49, 15)

    # all-atom H-bond data:
    df_cesar_aa_h_bonds = pd.read_excel(data,
                                        sheet_name=1)
    # some empty (NaN) rows and columns to filter out:
    for axis in [0, 1]:
        df_cesar_aa_h_bonds.dropna(axis=axis, how='all', inplace=True)
    assert df_cesar_aa_h_bonds.isna().sum().sum() == 0 # shape: (49, 6)

    # fuse Cesar's AA-MD data on the WT % columns
    df_cesar_aa = _merge_dfs(df_cesar_aa_enthalpy, df_cesar_aa_h_bonds)
    # the fused df shape should preserve rows
    # and sum columns (-3 for the WT % combo columns)
    assert df_cesar_aa.shape == (49, 18)
    # Prefix the feature columns with "AA_" to distinguish from
    # the other CG data
    df_cesar_aa = _add_df_col_prefix(df=df_cesar_aa,
                                     start_index=3,
                                     prefix="AA_")
    return df_cesar_aa


def _add_df_col_prefix(df, start_index: int, prefix: str):
    # add a prefix to a subset of
    # dataframe column names for clarity
    rename_dict = {}
    for old_col_name in df.columns[start_index:]:
        rename_dict[old_col_name] = f"{prefix}{old_col_name}"
    df = df.rename(columns=rename_dict)
    return df

def _merge_dfs(df1, df2):
    # a common dataframe merge scheme we use
    df = df1.merge(df2,
                   on=["WT% DEX",
                       "WT% PEO",
                       "WT% WATER"])
    return df


def feature_importance_consensus(pos_class_feat_imps: Sequence[npt.NDArray[np.float64]],
                                 feature_names: npt.NDArray,
                                 top_feat_count: int) -> Tuple[npt.NDArray, npt.NDArray[np.int64], int]:
    """
    Parameters
    ----------
    pos_class_feat_imps: a sequence of NumPy arrays; each NumPy array corresponds
                         to either a shape (n_records, n_features) collection of SHAP values
                         for a given ML model (values are for the positive class
                         selection), or to a reduced version of this data structure
                         like with random forest feature importances with shape
                         (n_features,)
    features_names: an array-like of strings of the features names of size ``n_features``
    top_feat_count: an integer representing the number of top features
                    to consider from each model when assessing the consensus

    Returns
    -------
    ranked_feature_names: array of feature names in descending order
                          of consensus importance (count) in the top
                          features per model
    ranked_feature_counts: array of counts (consensus occurrences) for
                           each feature in ``ranked_feature_names``
    num_input_models: int
    """
    num_input_models = len(pos_class_feat_imps)
    # calculate the mean absolute SHAP
    # values for each input ML model
    # OR simply the absolute values for already-reduced
    # feature importances
    processed_feat_imps = []
    for pos_class_imp_arr in pos_class_feat_imps:
        if np.atleast_2d(pos_class_imp_arr).shape[0] > 1:
            # haven't reduced across the records yet (like raw SHAP importances)
            processed_feat_imps.append(np.mean(np.absolute(pos_class_imp_arr), axis=0))
        else:
            processed_feat_imps.append(np.absolute(pos_class_imp_arr))
    # for each input ML model store the
    # top_feat_count feature names
    top_feat_data: dict[str, int] = defaultdict(int)
    for processed_feat_imp in processed_feat_imps:
        sort_idx = np.argsort(processed_feat_imp)[::-1]
        top_feature_names = feature_names[sort_idx][:top_feat_count]
        for top_feature_name in top_feature_names:
            top_feat_data[top_feature_name] += 1
    top_feat_data = dict(sorted(top_feat_data.items(),
                                key=lambda item: item[1],
                                reverse=True))
    ranked_feature_names = np.asarray(list(top_feat_data.keys()))
    ranked_feature_counts = np.asarray(list(top_feat_data.values()))
    return ranked_feature_names, ranked_feature_counts, num_input_models


def plot_feat_import_consensus(ranked_feature_names: npt.NDArray,
                               ranked_feature_counts: npt.NDArray[np.int64],
                               num_input_models: int,
                               top_feat_count: int,
                               fig_name: Optional[str] = "feat_imp_consensus.png"):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    y_pos = np.arange(ranked_feature_names.size)
    ax.barh(y_pos,
            (ranked_feature_counts/num_input_models) * 100)
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"% ML models where ranked in top {top_feat_count} features")
    ax.set_yticks(y_pos, labels=ranked_feature_names)
    ax.set_title(f"Feature importance consensus amongst {num_input_models} models")
    fig.tight_layout()
    fig.savefig(fig_name, dpi=300) # type: ignore
    return fig


def get_positive_shap_values(shap_values):
    # for the type handling here, see release 0.45.0 and
    # https://github.com/shap/shap/pull/3318
    if isinstance(shap_values, list):
        positive_class_shap_values = shap_values[1]
    else:
        if shap_values.ndim == 3:
            positive_class_shap_values = shap_values[:, :, 1]
        else:
            # XGBoost case?
            positive_class_shap_values = shap_values
    return positive_class_shap_values


def select_k_best_scores(X, y, k, metrics):
    res = []
    for metric in metrics:
        selector = SelectKBest(metric, k=k)
        selector.fit(X.to_numpy(), y)
        selector_feat_scores = selector.scores_
        assert selector_feat_scores.size == X.shape[1]
        res.append(selector_feat_scores)
    return res


@memory.cache
def build_lime_data(X, model):
    """
    LIME feature importances are calculated one record
    at a time, and spit out as a dictionary-like data
    structure, so we need to do a bit of work to get
    things in good shape for the consensus feature importance
    analysis.

    Parameters
    ----------
    X: should be the DataFrame containing the design matrix
    model: should be a pre-fit ML model for which feature importances
           are to be calculated for

    Returns
    -------
    feat_importances: the feature importances for each record
                      in a NumPy array with same shape as X.

    Notes
    -----
    Only desiged to work with tabular data.
    """
    check_is_fitted(model)
    out = np.empty(shape=X.shape, dtype=np.float64)
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(X.to_numpy(),
                                                            feature_names=X.columns)
    for index, row in tqdm(X.iterrows(),
                           total=X.shape[0],
                           desc="build LIME feature importance array"):
        exp = explainer_lime.explain_instance(X.to_numpy()[index],
                                              model.predict_proba,
                                              num_features=X.shape[1])
        exp_arr = np.asarray(list(exp.as_map().values())[0])
        # sort the feature importance scores to order them
        # alongside the columns (first score is for column 0, etc...)
        exp_arr = exp_arr[exp_arr[:, 0].argsort()]
        # discard the col indices and only keep the scores, to match other
        # feature importance approaches
        lime_local_scores = exp_arr[:, 1]
        assert lime_local_scores.size == X.shape[1]
        out[index, :] = lime_local_scores
    return out
