from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional, Any, Sequence

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from skimage import measure 
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances_argmin_min

def _safe_read_excel(
    path: Path, *, 
    sheet_name: str, 
    **kwargs: Any
) -> pd.DataFrame:
    """
    Safely reads a sheet from an Excel file, providing 
    a clear error if the file is missing.

    This function acts as a wrapper around `pandas.read_excel`. 
    It first verifies the existence of the file at the 
    specified path before attempting to read it. Any additional
    keyword arguments are passed directly to the underlying
    pandas function.

    Parameters
    ----------
    path : Path
        The file path to the Excel workbook.
    sheet_name : str
        The name of the sheet to read from the workbook.
    **kwargs : Any
        Additional keyword arguments to pass to `pandas.read_excel`.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the data from the specified Excel sheet.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    
    df = pd.read_excel(path, sheet_name=sheet_name, **kwargs)
    return df


def _axis_ranges(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    x_col: str,
    y_col: str,
    pad: int = 2,
) -> Tuple[list[int], list[int]]:
    """
    Calculates shared axis ranges for plotting two DataFrames.

    This utility function finds the maximum value for specified
    x and y columns across two separate DataFrames. It then 
    creates axis ranges from zero to this maximum value plus 
    a given padding.

    Parameters
    ----------
    df1 : pd.DataFrame
        The first DataFrame for comparison.
    df2 : pd.DataFrame
        The second DataFrame for comparison.
    x_col : str
        The name of the column representing the x-axis.
    y_col : str
        The name of the column representing the y-axis.
    pad : int, optional
        The padding to add to the maximum axis value. Defaults to 2.

    Returns
    -------
    Tuple[list[int], list[int]]
        A tuple containing two lists: the x-axis range [0, x_max] and the
        y-axis range [0, y_max].
    """
    x_max = max(df1[x_col].max(), df2[x_col].max()) + pad
    y_max = max(df1[y_col].max(), df2[y_col].max()) + pad
    return [0, int(x_max)], [0, int(y_max)]

class GMMWrapper:
    """
    A wrapper to use a GMM trained on feature data as a 
    classifier in composition space.

    This class bridges the gap between a GMM trained on 
    1D phase data and the 2D composition space of a phase 
    iagram. For any given composition point (x, y), it 
    finds the nearest experimental data point in the 
    composition space and uses that point's phase to predict
    a cluster label with the GMM.

    Attributes:
        gmm (GaussianMixture): The pre-trained Gaussian 
                               Mixture Model.
        x_comp (np.ndarray): The 2D array of composition
                             data (n_samples, 2).
        x_features (np.ndarray): The corresponding 
                                 feature vectors used for 
                                 GMM training.
    """

    def __init__(
        self,
        gmm: GaussianMixture,
        x_comp: np.ndarray,
        x_features: np.ndarray
    ):
        """
        Initializes the GMMWrapper.

        Parameters:
        ----------
            gmm (GaussianMixture): The trained GMM instance.
            x_comp (np.ndarray): The composition data array 
                                 (n_samples, 2).
            x_features (np.ndarray): The feature data array 
                                     (n_samples, 1).
        """
        self.gmm = gmm
        self.x_comp = x_comp
        self.x_features = x_features

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """
        Predicts cluster labels for new composition points.

        Parameters:
        -----------
            x_test (np.ndarray): An array of new composition
                                 points (n_test_samples, 2).

        Returns:
        -------
            np.ndarray: The predicted GMM cluster labels 
                        for the input points.
        """
        closest_indices, _ = pairwise_distances_argmin_min(
            x_test, 
            self.x_comp
            )
        feature_vectors = self.x_features[closest_indices]
        labels = self.gmm.predict(feature_vectors)
        return labels


def extract_boundary_from_contour(
    z: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    level: float = 0.5
) -> Optional[np.ndarray]:
    """
    Extracts the longest boundary contour from a grid at 
    a specified level.

    Parameters:
    ----------
        z (np.ndarray): The 2D grid of predicted values.
        xs (np.ndarray): The x-coordinates corresponding
                         to the grid columns.
        ys (np.ndarray): The y-coordinates corresponding 
                         to the grid rows.
        level (float): The contour level to extract.

    Returns:
    --------
        Optional[np.ndarray]: An array of (x, y) coordinates
                              for the boundary, or None if 
                              no contour is found.
    """
    contours = measure.find_contours(z, level)
    if not contours:
        return None
    longest_contour = max(contours, key=len)
    resolution_x, resolution_y = len(xs), len(ys)
    boundary_points = np.column_stack((
        np.interp(longest_contour[:, 1], 
                  np.arange(resolution_x), xs),
        np.interp(longest_contour[:, 0], 
                  np.arange(resolution_y), ys)
    ))
    return boundary_points

def _standardise_labels(
    cluster_labels: np.ndarray, x_comp: np.ndarray
) -> tuple[np.ndarray, dict[int, int]]:
    """
    Remap raw GMM labels so conventions never flip.

    *Standard convention used downstream*

    ---------  ----------------------------
    label=0    Two Phase   (triangle-up, turquoise)
    label=1    Single Phase (square, light-steel-blue)
    ---------  ----------------------------

    Parameters
    ----------
    cluster_labels :
        Original labels from :pyclass:`sklearn.mixture.GaussianMixture`.
    x_comp :
        Composition coordinates associated with each label.

    Returns
    -------
    std_labels :
        Relabelled array following the convention above.
    label_map :
        Dict mapping raw → standard labels; apply to any
        future predictions (`z` grids, etc.).
    """
    centroids = {
        lbl: x_comp[cluster_labels == lbl].mean(axis=0)
        for lbl in np.unique(cluster_labels)
    }
    distances = {lbl: np.linalg.norm(c) for lbl, c in centroids.items()}

    near_lbl = min(distances, key=lambda lbl: float(distances[lbl]))
    far_lbl  = max(distances, key=lambda lbl: float(distances[lbl]))

    label_map = {far_lbl: 0, near_lbl: 1}
    std_labels = np.vectorize(label_map.get)(cluster_labels)
    return std_labels, label_map

def _set_axis_style(ax, xrange: list[int] | None, yrange: list[int] | None):
    """
    Apply consistent styling *and* force identical tick-spacing & integer labels.

    - step = round(max(xrange_span, yrange_span)/5)
    - both axes MajorLocator(step)
    - both axes FormatStrFormatter('%.0f')
    """
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    ax.set_aspect("equal", adjustable="box")

    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(
        direction="in",
        length=5,
        width=2,
        which="both",
        top=True,
        right=True,
        labelsize=24,
    )

def plot_gmm_decision_regions(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    phase_col: str,
    ax,
    xrange: Sequence[float],
    yrange: Sequence[float],
    n_components: int = 2,
    random_state: int = 42,
    region_colors: Optional[list[str]] = None,
    boundary_color: str = "red",
    resolution: int = 200,
    decision_alpha: float = 0.3,
    plot_regions: bool = True
) -> Tuple[GaussianMixture, np.ndarray, Optional[np.ndarray]]:
    """
    Trains a GMM and creates contour traces for phase 
    regions and boundaries.

    Parameters:
    -----------
        df (pd.DataFrame): DataFrame with composition and
                           phase data.
        x_col (str): Name of the column for the x-axis 
                     component.
        y_col (str): Name of the column for the y-axis 
                     component.
        phase_col (str): Name of the column containing 
                         phase information.
        xrange (list[int]): X-axis composition range.
        yrange (list[int]): Y-axis composition range. 
        n_components (int): Number of GMM components.
        random_state (int): Seed for reproducibility.
        region_colors (Optional[list[str]]): Colors for the 
                                             phase regions.
        boundary_color (str): Color for the boundary line.
        resolution (int): Grid resolution for the contour plot.
        decision_alpha (float): Opacity of the filled regions.
        plot_regions (bool): Whether to plot the filled regions.

    Returns:
    --------
        Tuple: Containing the GMM model, cluster labels, 
               boundary points, and a list of Plotly contour
               traces.
    """
    df_local = df.copy()
    x_features = df_local[phase_col].to_numpy().reshape(-1, 1)

    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    raw_labels = gmm.fit_predict(x_features)

    x_comp = df_local[[x_col, y_col]].to_numpy()
    std_labels, label_map = _standardise_labels(raw_labels, x_comp)
    wrapper = GMMWrapper(gmm, x_comp, x_features)
    x_start, x_stop = map(float, xrange)
    y_start, y_stop = map(float, yrange)
    xs = np.linspace(x_start, x_stop, resolution)
    ys = np.linspace(y_start, y_stop, resolution)
    xx, yy = np.meshgrid(xs, ys)
    z_raw = wrapper.predict(np.c_[xx.ravel(), yy.ravel()])
    z = np.vectorize(label_map.get)(z_raw).reshape(xx.shape)

    if region_colors is None:
        region_colors = ["aquamarine", "lightsteelblue"]

    if plot_regions:
        ax.contourf(
            xx,
            yy,
            z,
            levels=[-0.5, 0.5, 1.5],
            colors=region_colors,
            alpha=decision_alpha,
        )

    ax.contour(
        xx,
        yy,
        z,
        levels=[0.5],
        colors=boundary_color,
        linewidths=3,
    )
    proxy_artist = Line2D(
        [0], 
        [0], 
        label="Decision Boundary", 
        color=boundary_color, 
        linewidth=3
    )
    ax.legend(handles=[proxy_artist])

    boundary_points = extract_boundary_from_contour(z, xs, ys, level=0.5)
    return gmm, std_labels, boundary_points

def plot_gmm_composition_phase(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    phase_col: str,
    ax,
    point_cmap: Optional[Sequence[str]] = None,
) -> None:
    """
    Trains a GMM and creates a scatter trace for data points.

    Optionally creates region traces as well, mimicking the 
    original script's logic.

    Parameters:
    -----------
        df (pd.DataFrame): DataFrame with composition and 
                           phase data.
        x_col (str): Name of the column for the x-axis 
                     component.
        y_col (str): Name of the column for the y-axis 
                     component.
        phase_col (str): Name of the column containing 
                         phase information.
        xrange (list[int]): X-axis composition range.
        yrange (list[int]): Y-axis composition range. 
        n_components (int): Number of GMM components.
        random_state (int): Seed for reproducibility.
        region_colors (Optional[list[str]]): Colors for the 
                                             phase regions.
        resolution (int): Grid resolution for any contour plots.
        plot_regions (bool): Whether to plot the filled regions.
        point_cmap (Optional[list[str]]): Colormap for the
                                          scatter points.

    Returns:
    -------
        None
    """
    x_features = df[phase_col].to_numpy().reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42)
    raw_labels = gmm.fit_predict(x_features)

    x_comp = df[[x_col, y_col]].to_numpy()
    std_labels, _ = _standardise_labels(raw_labels, x_comp)
    if point_cmap is None:
        point_cmap = ("aquamarine", "lightsteelblue")
    mask_two = std_labels == 0
    ax.scatter(
        x_comp[mask_two, 0], x_comp[mask_two, 1],
        marker="^", c=point_cmap[0],
        s=120, edgecolors="black", linewidths=1,
        label="Two Phase (Experiment)",                       
    )

    mask_single = std_labels == 1
    ax.scatter(
        x_comp[mask_single, 0], x_comp[mask_single, 1],
        marker="s", c=point_cmap[1],
        s=120, edgecolors="black", linewidths=1,
        label="Single Phase (Single Phase)",
    )