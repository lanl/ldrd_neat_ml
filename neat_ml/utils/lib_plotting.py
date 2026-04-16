from . import figure_utils
import os
from pathlib import Path
from typing import Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import json
import warnings
import math
import logging
from matplotlib import rcParams
from matplotlib.axes import Axes
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
import itertools

# try setting plot font to ``Arial``, if installed, 
# otherwise default to standard matplotlib font
rcParams['font.sans-serif'] = ["Arial"]
rcParams['font.family'] = "sans-serif"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_parameters_from_json(path: Path) -> dict[str, float]:
    """
    Read and validate the model fit parameters file.

    The parameters were obtained from equation 1 of the
    following literature:

    Silvério, Sara C., et al. "Effect of aqueous two-phase 
    system constituents in different poly (ethylene 
    glycol)-salt phase diagrams." Journal of Chemical & 
    Engineering Data 57.4 (2012): 1203-1208.
    
    Parameters:
    -----------
    path : Path
        path from which to load parameters

    Returns
    -------
    params : dict[str, float]
        A dictionary containing the validated model parameters.
    """
    REQUIRED_KEYS = {"MODEL_A", "MODEL_B", "MODEL_C"}
    
    with open(path, "r") as fh:
        params = json.load(fh)

    missing_keys = REQUIRED_KEYS - params.keys()
    if missing_keys:
        raise KeyError(f"{path} is missing required keys: {missing_keys}")

    for key, value in params.items():
        if not isinstance(value, (int, float)):
            raise TypeError(f"Parameter '{key}' must be a number, but got {type(value).__name__}.")

    return params

def titration_diagram(
    file_path: Path,
    x_col: str,
    y_col: str,
    phase_col: str,
    xrange: list[int],
    yrange: list[int],
    output_path: Path,
) -> None:
    """
    Load data and plot two-phase scatter diagram.

    Parameters
    ----------
    file_path : Path
        Path to the input CSV file.
    x_col : str
        Name of the column for x-axis values.
    y_col : str
        Name of the column for y-axis values.
    phase_col : str
        Column name for phase labels (0 or 1).
    xrange : list[int]
        [min, max] range for the x-axis.
    yrange : list[int]
        [min, max] range for the y-axis.
    output_path : Path
        Path to save the output PNG image.
    """
    df = pd.read_csv(file_path)

    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)

    scatter_cfg: list[dict[str, Any]] = [
        {
            "val": 0,
            "color": "dodgerblue",
            "marker": "s",
            "label": "Single Phase",
            "ms": 20,
        },
        {
            "val": 1,
            "color": "#FF8C00",
            "marker": "^",
            "label": "Two Phase",
            "ms": 20,
        },
    ]

    for cfg in scatter_cfg:
        mask: pd.Series = df[phase_col] == cfg["val"]
        ax.scatter(
            df.loc[mask, x_col],
            df.loc[mask, y_col],
            c=cfg["color"],
            marker=cfg["marker"],
            s=180,
            edgecolors="black",
            linewidths=1,
            label=cfg["label"],
        )

    # Mihee requests changing ``PEO`` to ``PEG`` in axis labels
    # perform programmatically here vs. modifying all input files
    _, y_col = figure_utils.rename_df_columns(df, y_col) 
    ax.set_xlabel(x_col, fontsize=46, fontweight="bold", labelpad=20)
    ax.set_ylabel(y_col, fontsize=46, fontweight="bold", labelpad=20)
    figure_utils._set_axis_style(ax, xrange, yrange)

    ax.legend(loc="upper right", framealpha=0.8, fontsize=34)
    
    dest_dir = os.path.dirname(output_path)
    if dest_dir and not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    fig.savefig(
        output_path, 
        bbox_inches="tight",
        pad_inches=1.0
    )
    plt.close(fig) 
    logger.info(f"Plot saved successfully to {output_path}")

def plot_two_scatter(
    csv1_path: Path,
    csv2_path: Path,
    output_path: Path,
    xlim: Optional[list[int]]=None,
    ylim: Optional[list[int]]=None,
) -> None:
    """
    Read two CSV files and plot styled scatter comparison.

    Parameters
    ----------
    csv1_path : Path
        Path to Titration.CSV file
    csv2_path : Path
        Path to TECAN.CSV file
    output_path : Path
        Path to save the output image.
    xlim : Optional[list[float]]
        [min, max] range for x-axis.
    ylim : Optional[list[float]]
        [min, max] range for y-axis.
    """

    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)

    x_col = df2.columns[0]
    y_col = df2.columns[1]

    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)

    scatter_cfg:  list[dict[str, Any]] = [
        {"df": df1, "color": "purple", "label": "Turbidimetric Titration"},
        {"df": df2, "color": "yellow", "label": "Computational Pipeline"},
    ]

    for cfg in scatter_cfg:
        data: pd.DataFrame = cfg["df"]
        ax.scatter(
            data[x_col],
            data[y_col],
            c=cfg["color"],
            marker="^",
            s=300,
            edgecolors="black",
            linewidths=1,
            label=cfg["label"],
        )
    
    # Mihee requests changing ``PEO`` to ``PEG`` in axis labels
    # perform programmatically here vs. modifying all input files
    _, y_col = figure_utils.rename_df_columns(df1, y_col) 
    _, x_col = figure_utils.rename_df_columns(df1, x_col)
    ax.set_xlabel(x_col, fontsize=46, fontweight="bold", labelpad=20)
    ax.set_ylabel(y_col, fontsize=46, fontweight="bold", labelpad=20)
    figure_utils._set_axis_style(ax, xlim, ylim)
    
    ax.legend(loc="upper right", framealpha=0.8, fontsize=30)

    dest_dir = os.path.dirname(output_path)
    if dest_dir and not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    fig.savefig(
        output_path, 
        bbox_inches="tight",
        pad_inches=1.0
    )
    plt.close(fig)
    logger.info(f"Plot saved successfully to {output_path}")

def plot_phase_diagram(
    file_path: Path,
    x_col: str,
    y_col: str,
    phase_col: str,
    output_path: Path,
    xrange: Optional[list[int]]=None,
    yrange: Optional[list[int]]=None,
    json_path: Optional[Path]=None,
    binodal_curve: bool=False,
    model_boundary: bool=False,
    pred_phase_col: Optional[str]=None,
) -> None:
    """
    Generate and save a phase diagram plot from composition data.

    Parameters
    ----------
    file_path : Path
        Path to the CSV input file.
    json_path: Optional[Path]
        Path to the .json file for model parameters
    x_col : str
        Column name for x-axis values.
    y_col : str
        Column name for y-axis values.
    phase_col : str
        Column name for phase labels.
    xrange : Optional[list[int]]
        Two-element list [min, max] for x-axis range.
    yrange : Optional[list[int]]
        Two-element list [min, max] for y-axis range.
    output_path : Path
        Path to save the output plot image.
    binodal_curve: bool
        Option to plot the binodal curve of fit for
        describing two-phase aqueous sytems as taken
        from Silverio et al.
    model_boundary: bool
        option to plot the phase boundary as determined
        by the trained ML model.
    pred_phase_col: str
        with ``model_boundary``, the dataframe column
        containing the model-predicted phase labels
    """
    df = pd.read_csv(file_path)

    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)

    _, _, boundary_exp = figure_utils.plot_gmm_decision_regions(
        df,
        x_col,
        y_col,
        phase_col,
        ax=ax,
        xrange=xrange,
        yrange=yrange,
        n_components=2,
        random_state=42,
        boundary_color="red",
        resolution=200,
        decision_alpha=1,
        plot_regions=True,
        region_colors=("lightsteelblue", "aquamarine"),
    )

    boundary_pred = None
    if model_boundary and pred_phase_col is not None:
        _, _, boundary_pred = figure_utils.plot_gmm_decision_regions(
            df,
            x_col,
            y_col,
            pred_phase_col,
            ax=ax,
            xrange=xrange,
            yrange=yrange,
            n_components=2,
            random_state=42,
            boundary_color="blue",
            resolution=200,
            decision_alpha=1,
            plot_regions=False,
            region_colors=("lightsteelblue", "aquamarine"),
            decision_boundary_width=2,
        )
            
    # if provided, use the ML predictions for plotting
    if pred_phase_col is not None:
        comp_phase_col = pred_phase_col
        plot_marker = "Model"
    else:
        comp_phase_col = phase_col
        plot_marker = "Experiment"

    figure_utils.plot_gmm_composition_phase(
        df,
        x_col,
        y_col,
        comp_phase_col,
        ax=ax,
        point_cmap=("#FF8C00", "dodgerblue"),
    )
    _, x_col = figure_utils.rename_df_columns(df, x_col)

    handles = [
        Patch(facecolor="lightsteelblue", edgecolor="black",
              label="Two-Phase (Experiment)"),
        Patch(facecolor="aquamarine", edgecolor="black",
              label="Single-Phase (Experiment)"),
        Line2D([0], [0], marker="^", color="w",
               markerfacecolor="#FF8C00", markeredgecolor="black",
               markersize=20, label=f"Two-Phase ({plot_marker})"),
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor="dodgerblue", markeredgecolor="black",
               markersize=20, label=f"Single-Phase ({plot_marker})"),
    ]
    if not model_boundary:
        handles.append(
            Line2D([0], [0], color="red", lw=3, label="Decision Boundary")
        )
    
    # plot the binodal curve describing the behavior of aqueous two-phase systems
    # as described in Silverio et al., equation (1), [dx.doi.org/10.1021/je2012549] 
    if binodal_curve and xrange is not None and yrange is not None:
        # require ``json_path`` when ``binodal_curve == True``
        if json_path is None:
            raise ValueError("Must provide ``json_path`` when plotting ``binodal_curve``.")
        model_vars = load_parameters_from_json(json_path)

        model_a: float = model_vars["MODEL_A"]
        model_b: float = model_vars["MODEL_B"]
        model_c: float = model_vars["MODEL_C"]
        raw_x = np.linspace(xrange[0], xrange[1], 500, dtype=float)
        frac_x = raw_x / 100.0
        model_y = model_a * np.exp(model_b * np.sqrt(frac_x) - model_c * frac_x**3) * 100.0

        ax.plot(
            raw_x,
            model_y,
            color="black",
            linewidth=2.5,
            label=f"Model fit: a={model_a:.3f}, b={model_b:.2f}, c={model_c:.0f}",
        )
        handles.append(
            Line2D([0], [0], color="black", lw=2.5, label="Binodal Fit (Silverio et al.)")
        )
    
    if model_boundary:
        if boundary_exp is not None:
            handles.append(Line2D([],[],color="red",lw=3,
                    label="Decision Boundary (Experiment)"
                )
            )
        if boundary_pred is not None:
            handles.append(Line2D([],[],color="blue",lw=3,
                    label="Decision Boundary (Model)"
                )
            )

    ax.legend(
        handles=handles, 
        bbox_to_anchor=(0.5, -0.2),
        loc="upper center", 
        framealpha=0.8, 
        fontsize=32,
        ncol=3,
    )

    # Mihee requests changing ``PEO`` to ``PEG`` in axis labels
    # perform programmatically here vs. modifying all input files
    _, y_col = figure_utils.rename_df_columns(df, y_col) 
    ax.set_xlabel(x_col, fontsize=46, fontweight="bold", labelpad=20)
    ax.set_ylabel(y_col, fontsize=46, fontweight="bold", labelpad=20)
    figure_utils._set_axis_style(ax, xrange, yrange)

    dest_dir = os.path.dirname(output_path)
    if dest_dir and not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    fig.savefig(
        output_path, 
        bbox_inches="tight",
        pad_inches=1.0
    )
    plt.close(fig)
    logger.info(f"Plot saved successfully to {output_path}")


def make_titration_figures(
    csv_dir: Path,
    out_dir: Path
) -> None:
    """
    Processes a CSV file to generate titration plots from each sheet.

    This function generates titration phase diagram as a PNG image.

    The first row of each CSV file is expected to contain the titles for the
    x-axis, y-axis, and phase columns, respectively. The resulting
    PNG files are named after the CSV file and are saved in the
    specified output directory.

    Parameters
    ----------
    csv_dir : Path
        The file path to the input csv files containing the titration data.
    out_dir : Path
        The path to the directory where the output CSV files and PNG
        images will be saved.
    """

    for csv_path in sorted(csv_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        x_title, y_title, phase_title = df.columns[:3]
        x_max = int(df[x_title].max())
        y_max = int(df[y_title].max())
        x_rng = [0, x_max + math.ceil(0.1 * x_max)]
        y_rng = [0, y_max + math.ceil(0.1 * y_max)]

        png_path = out_dir / f"{csv_path.stem}_Titration_Phase_Diagram.png"

        titration_diagram(
            file_path=csv_path,
            x_col=x_title,
            y_col=y_title,
            phase_col=phase_title,
            xrange=x_rng,
            yrange=y_rng,
            output_path=png_path,
        )

def make_binodal_comparison_figures(
    csv_dir: Path,
    out_dir: Path
) -> None:
    """
    Generates Titrate-vs-TECAN comparison plots for every paired dataset.

    This function scans CSV files in the directory with names matching a
    specific pattern (e.g., "A_Titrate", "A_TECAN"). It groups these CSV
    by their common identifier (e.g., "A").

    For each identified pair, it reads the data, assuming the first row
    contains the x and y titles. Finally, it generates a scatter plot 
    comparing the two datasets and saves it as a PNG image.

    Parameters
    ----------
    csv_dir : Path
        The file path to the input CSV file directory.
    out_dir : Path
        The path to the directory where the output CSV files and PNG
        images will be saved.
    """

    titrate: dict[str, Path] = {}
    tecan: dict[str, Path] = {}
    
    # fixed axis ranges requested by Mihee
    # based on reasonable assumptions of
    # experimental conditions.
    axis_ranges = {
        "Dextran 500 kg/mol (wt%)": [0, 12],
        "PEO 20 kg/mol (wt%)": [0, 4],
        "Sodium Citrate (wt%)": [0, 18], 
        "PEO 8 kg/mol (wt%)": [0, 40],
        "Dextran 10 kg/mol (wt%)": [0, 14],
        "PEO 10 kg/mol (wt%)": [0, 14],
    }
     
    file_suffix = ["1st", "2nd"]
    for suff in file_suffix:
        for p in csv_dir.glob("*_Titrate.csv"):
            titrate[p.stem.replace("_Titrate", "")] = p  # strip '_Titrate'
        for p in csv_dir.glob(f"*_TECAN_{suff}.csv"):
            tecan[p.stem.replace(f"_TECAN_{suff}", "")] = p  # strip '_TECAN'
        for ds_id in sorted(titrate):
            csv_tit = titrate[ds_id]
            csv_tec = tecan[ds_id]

            df_tit = pd.read_csv(csv_tit)
            x_col, y_col = df_tit.columns[:2]
            df_tit, x_col = figure_utils.rename_df_columns(df_tit, x_col)
            png_path = out_dir / f"Figure_6_{ds_id}_Binodal_Comparison_{suff}.png"

            plot_two_scatter(
                csv1_path=csv_tit,
                csv2_path=csv_tec,
                output_path=png_path,
                xlim=axis_ranges[x_col],
                ylim=axis_ranges[y_col],
            )

def make_phase_diagram_figures(
    csv_dir: Path, 
    out_dir: Path,
    phase_cols: tuple[str, str]
) -> None:
    """Iterates through CSV files to produce two diagrams per file.

    This function searches the specified csv_dir for all files ending
    in .csv. For each file found, it reads the data and attempts to
    generate up to two phase diagrams.

    It assumes that the x and y data are in the 4th and 5th columns,
    respectively. It then checks for the existence of specific phase
    columns (defined in phase_cols) to create a diagram for each
    phase found.

    Parameters
    ----------
    csv_dir : Path
        The path to the directory containing the input CSV files.
    out_dir : Path
        The path to the directory where the output PNG images will be saved.
    phase_cols : tuple[str, str]
        The tuple of phase separation column names used for phase diagram
    """
    csv_files = sorted(p for p in csv_dir.iterdir() if p.suffix.lower() == ".csv")
    if not csv_files:
        raise ValueError(f"No CSV files found in directory: {csv_dir}")

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)

        try:
            x_title, y_title = df.columns[3:5]
        except ValueError as exc:
            raise ValueError(
                f"CSV {csv_path} must have at least five columns; got {len(df.columns)}"
            ) from exc

        x_rng = [0, int(df[x_title].max()) + 1]
        y_rng = [0, int(df[y_title].max()) + 1]
        ds_id = csv_path.stem

        for tag, phase_col in zip(("1st", "2nd"), phase_cols):
            if phase_col not in df.columns:
                warnings.warn(
                    "Dataframe missing phase columns. Skipping.",
                    UserWarning,
                )
                continue
            png_path = out_dir / f"{ds_id}_Phase_Diagram_{tag}.png"
            plot_phase_diagram(
                file_path=csv_path,
                x_col=x_title,
                y_col=y_title,
                phase_col=phase_col,
                xrange=x_rng,
                yrange=y_rng,
                output_path=png_path,
            )

def plot_figures(
    titration_csv_dir: Path,
    binodal_csv_dir: Path,
    csv_phase_dir: Path,
    out_dir: Path,
    mat_model_csv: Path,
    mat_model_png: Path,
    json_path: Path,
    phase_cols: tuple[str, str],
    xrange: list[int],
    yrange: list[int]
) -> None:
    """
    Run all three figure-generation pipelines.

    The routine is a thin orchestrator: it delegates the heavy
    lifting to the specialised helper functions defined in
    this module (make_titration_figures, make_binodal_comparison_figures,
    make_phase_diagram_figures, and binodal_model).

    Parameters
    ----------
    titration_csv_dir : pathlib.Path
        CSV files containing individual titration datasets
    binodal_csv_dir : pathlib.Path
        CSV files containing paired “Titrate/TECAN” binodal data
    csv_phase_dir : pathlib.Path
        Directory containing one or more CSV files with composition/phase data
    out_dir : pathlib.Path
        Destination directory for all generated artifacts (PNGs and any
        intermediate CSV exports).
    mat_model_csv : pathlib.Path
        Single CSV sheet that feeds the binodal model comparison plot
    mat_model_png : pathlib.Path
        Output filename for the binodal model phase diagram
    json_path : pathlib.Path, default JSON_PATH
        JSON file containing the calibrated model parameters
        MODEL_A, MODEL_B, and MODEL_C.
    phase_cols: tuple[str, str]
        Tuple of strings to represent the phase column names 
    xrange: list[int]
        X-axis range for the plot
    yrange: list[int]
        Y-axis range for the plot
    """


    make_titration_figures(titration_csv_dir, out_dir)
    make_binodal_comparison_figures(binodal_csv_dir, out_dir)
    make_phase_diagram_figures(csv_phase_dir, out_dir, phase_cols)

    plot_phase_diagram(
        file_path=mat_model_csv,
        json_path=json_path,
        x_col="Sodium citrate (wt%)",
        y_col="PEO 8 kg/mol (wt%)",
        phase_col="Phase_Separation_2nd",
        xrange=xrange,
        yrange=yrange,
        output_path=mat_model_png,
        binodal_curve=True,
    )


def _is_inside_hull(points: np.ndarray, hull: ConvexHull) -> np.ndarray:
    """
    Checks if points are inside a convex hull using its linear equations.
    Equations are in the form: [normal_x, normal_y, offset]

    Parameters:
    -----------
    points: np.ndarray
        points to check if inside the convex hull 
    hull: ConvexHull
        the hull object containing the equations describing the
        hyperplane of the convex hull

    Returns:
    --------
    np.ndarray
        the boolean array of whether a given point is within
        the convex hull
    """
    # points: (N, 2), equations: (M, 3)
    # Result of dot product is (N, M). Point is inside if all values <= 1e-12
    return np.all(points @ hull.equations[:, :-1].T + hull.equations[:, -1] <= 1e-12, axis=1)


def _get_hull_overlaps(
    df: pd.DataFrame,
    features: list[str],
    label_col: str,
) -> pd.DataFrame:
    """
    Calculate the number of points contained in the overlapping region
    of the convex hulls of the input features.

    Parameters:
    -----------
    df: pd.DataFrame
        the input dataframe containing all of the feature values
    features: list[str]
        the feature names to compare
    label_col: str 
        the column name containing the ground-truth labels

    Returns:
    --------
    results: pd.DataFrame
        a dataframe containing a summary of the overlapping points
        between the input feature pair
    """
    class_labels = df[label_col].dropna().unique()
    results = []

    for x_feat, y_feat in itertools.combinations(features, 2):
        for class1, class2 in itertools.combinations(class_labels, 2):

            pts1 = df[df[label_col] == class1][[x_feat, y_feat]].dropna().values
            pts2 = df[df[label_col] == class2][[x_feat, y_feat]].dropna().values
            
            # get all points from both groups
            all_points = np.vstack([pts1, pts2])
            
            if len(pts1) > 2 and len(pts2) > 2:
                hull1 = ConvexHull(pts1)
                hull2 = ConvexHull(pts2)
                
                # check which points are in hull overlap
                in_hull1 = _is_inside_hull(all_points, hull1)
                in_hull2 = _is_inside_hull(all_points, hull2)
                in_both = in_hull1 & in_hull2
                
                overlap_count = np.sum(in_both)
                
                pts1_in_overlap = np.sum(_is_inside_hull(pts1, hull1) & _is_inside_hull(pts1, hull2))
                pts2_in_overlap = np.sum(_is_inside_hull(pts2, hull1) & _is_inside_hull(pts2, hull2))
                
                results.append({
                    'feature_x': x_feat,
                    'feature_y': y_feat,
                    'comparison': f"{class1} vs {class2}",
                    'points_1_in_overlap': pts1_in_overlap,
                    'points_2_in_overlap': pts2_in_overlap,
                    'overlap_count': overlap_count
                })

    return pd.DataFrame(results)


def _draw_hull(
    x: np.ndarray,
    y: np.ndarray,
    color: str, 
    ax: Axes
) -> None:
    """
    Draw the convex hull of the given feature points
    on the scatterplot of features
    
    Parameters:
    -----------
    x: np.ndarray
        the scatterplot x coordinates
    y: np.ndarray
        the scatterplot y coordinates
    color: str
        the plot color to use for drawing the hull
    ax: Axes
        the figure axes for plotting
    """
    points = np.column_stack((x, y))
    points = points[~np.isnan(points).any(axis=1)]
    
    if len(points) > 2:
        hull = ConvexHull(points)
        vertices = points[hull.vertices]
        
        vertices = np.vstack([vertices, vertices[0]])
        
        ax.fill(vertices[:, 0], vertices[:, 1], 
                facecolor=color, alpha=0.15, edgecolor=color, linewidth=2)


def generate_feature_scatterplots(
    input_df: pd.DataFrame,
    label_col: str,
    out_path: Path,    
    plot_cols: Optional[list[str]] = None,
) -> None:
    """
    Plot pairwise feature scatterplots of input features
    from a pandas dataframe with user provided feature names
    or by determining the top-3 most separated feature pairs
    within the top-10 features from the FIC plot.

    Parameters:
    -----------
    input_df: pd.DataFrame
        A pandas dataframe containing feature columns
        and rows of data points for plotting
    label_col: str
        The row of the dataframe for plotting the data points
        by their class label.
    out_path: Path
        Path for saving the pairplot
    plot_cols: list[str] | None
        User provided feature columns to include in the plot.
        If no columns provided, the top n features from the FIC
        plot will be used to generate pairwise feature scatterplots
        by finding the top-3 most separated features in terms of the
        distance between their centroids
    """
    # the top 10 features from the feature importance
    # consensus plot resulting from training the ML
    # classifier on the PEO10k/Dex10k composition.
    # as shown in the Manuscript figure S1.
    top_n_feats = [
        "median_nnd_std",
        "graph_num_components_std",
        "median_blob_area_min",
        "graph_avg_neighbor_distance_median",
        "median_blob_radius_min",
        "median_nnd_max",
        "num_blobs_std",
        "graph_num_nodes_std",
        "mean_blob_radius_min",
        "median_voronoi_area_min",
    ]
    # corresponding descriptions in plain english of the
    # code variable/feature names above.
    top_n_feat_names = [
        "standard deviation of the nearest neighbor distances",
        "standard deviation of the graph component counts",
        "minimum of the median blob areas",
        "median of the average graph neighbor distances",
        "minimum of the median blob radii",
        "maximum of the median nearest neighbor distances",
        "standard deviation of the number of blobs",
        "standard deviation of the number of graph nodes",
        "minimum of the mean blob radii",
        "minimum of the median voronoi areas",
    ]
    # check that the appropriate columns contained within `input_df`
    if plot_cols is not None:
        if len(plot_cols) < 2:
            raise ValueError(
                "Need at least two feature columns to generate pairwise scatterplots."
            )
        check_cols = [label_col] + plot_cols
        plot_feats = plot_cols
    else:
        check_cols = [label_col] + top_n_feats
        plot_feats = top_n_feats
    if not set(check_cols).issubset(input_df.columns):
        raise ValueError("Required columns are not present in ``input_df``")
    
    input_df = input_df[check_cols]
    # get the top-n feature columns
    agg_feats = input_df[plot_feats]
    if plot_cols is None:
        # normalize values with z-score
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(agg_feats)
        scaled_df = pd.DataFrame(scaled_data, columns=agg_feats.columns)
        # get the centroids of the feature columns
        centroids = scaled_df.mean()
        # compute the pairwise distances between the feature columns
        dist_array = pdist(centroids.to_numpy().reshape(-1, 1))
        output_df = pd.DataFrame(
            squareform(dist_array),
            index=centroids.index,
            columns=centroids.index,
        )
        # find the top-n largest distances
        all_pairs = output_df.unstack()
        top_n_pairs = all_pairs[
            all_pairs > 0].sort_values(ascending=False).head(10) # type: ignore[call-overload, index]
        top_n_out = top_n_pairs.iloc[::2].head(2)
        top_n_unique = list(top_n_out.index.to_series().explode().unique())
        top_n_idx = [top_n_feats.index(x) for x in top_n_unique]
        top_n_unique_names = [top_n_feat_names[i] for i in top_n_idx]
    else:
        top_n_unique = plot_cols
        top_n_unique_names = plot_cols
    input_df[label_col] = input_df[label_col].map({1: 'Two-Phase', 0: 'One-Phase'})
    stats_df = _get_hull_overlaps(input_df, top_n_unique, label_col)
    palette = ['#1f77b4', '#ff7f0e']

    # iterate through every unique pair of features
    for i, j in itertools.combinations(range(len(top_n_unique)), 2):
        x_feat = top_n_unique[i]
        y_feat = top_n_unique[j]
        x_name = top_n_unique_names[i]
        y_name = top_n_unique_names[j]
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        for i, cat in enumerate(input_df[label_col].unique()):
            subset = input_df[input_df[label_col] == cat].dropna(subset=[x_feat, y_feat])

            if len(subset) > 2:
                ax.scatter(subset[x_feat], subset[y_feat],
                            color=palette[i], alpha=0.6, label=cat, s=20)

                _draw_hull(subset[x_feat], subset[y_feat], color=palette[i], ax=ax)
        overlap_count = stats_df[(stats_df["feature_x"] == x_feat) & (stats_df["feature_y"] == y_feat)]["overlap_count"].item()
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_title(f"Feature Comparison Plot\n Number of Overlapping Points: {overlap_count}")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout()
        
        plt.savefig(out_path / f"hull_{x_feat}_vs_{y_feat}.png", dpi=300)
        plt.close()
