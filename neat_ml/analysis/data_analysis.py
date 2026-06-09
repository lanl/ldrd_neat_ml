import re
import warnings
from pathlib import Path
from typing import Any, Optional, Sequence, Union, Literal

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import KDTree, Voronoi, Delaunay
import logging
from tqdm.auto import tqdm
from pyarrow.lib import ArrowInvalid  # type: ignore[import-untyped]
import ast

__all__ = [
    "full_analysis"
]

log = logging.getLogger(__name__)


def _std_agg(x):
    return x.std(ddof=0)


def _merge_composition_data(
    summary_df: pd.DataFrame,
    composition_df: pd.DataFrame,
    *,
    cols_to_add: Sequence[str],
    merge_key: str,
) -> pd.DataFrame:
    """
    Merges a summary DataFrame with an external composition dataframe.

    Performs a left merge to add columns from the composition 
    dataframe to the summary metrics table based on a shared key.

    Parameters
    ----------
    summary_df : pd.DataFrame
        The per-image or aggregated metrics table, must contain 
        merge_key.
    composition_df : pd.DataFrame
        The external table with composition data.
    cols_to_add : Sequence[str]
        A sequence of columns from composition_df to add to summary_df.
    merge_key : str
        The column name to use as the merge key.

    Returns
    -------
    pd.DataFrame
        The merged DataFrame with the added composition columns.
    """
    if merge_key not in summary_df.columns:
        raise ValueError(f"Merge key '{merge_key}' not found in summary_df.")
    missing_cols = [c for c in (merge_key, *cols_to_add) if c not in composition_df.columns]
    if missing_cols:
        raise ValueError(f"Columns {missing_cols} not found in composition_df.")

    return summary_df.merge(
        composition_df[[merge_key, *cols_to_add]], on=merge_key, how="left"
    )

def _parse_filename(
    fname: str,
    method: Literal["BubbleSAM", "OpenCV"]
) -> dict[str, Any]:
    """
    Parses a bubblesam or opencv detection-generated filename to extract metadata.

    Parameters
    ----------
    fname : str
        The filename, e.g., 'offset -1_center_A1_Bf_Raw_uuid_bubble_data.parquet.gzip'.
    method : Literal["BubbleSAM", "OpenCV"]
        The method used for detection (bubblesam or opencv)

    Returns
    -------
    dict[str, Any]
        A dictionary with 'UniqueID', 'Class', 'Offset', 'Position', and 'Label',
        or an empty dictionary if the pattern does not match.
    """
    if method.lower() == "bubblesam":
        tag = "masks_filtered"
    else:
        tag = "bubble_data"

    _RE = re.compile(
        r"offset\s*(-?\d+)_"
        r"(bottom|top|left|right|center)_"
        r"([A-Z]\d+)_"
        r".+?_"
        r"(Bf|Ph)_Raw_"
        fr"([0-9a-f\-]+)_{tag}\.parquet\.gzip",
        re.IGNORECASE | re.VERBOSE,
    )
    match = _RE.match(fname)
    if not match:
        return {}
    offset, pos, label, cls, uid = match.groups()
    return {
        "UniqueID": uid,
        "Class": cls,
        "Offset": int(offset),
        "Position": pos,
        "Label": label,
    }


def _load_df(
    parquet_path: Path,
    method: Literal["BubbleSAM", "OpenCV"],
) -> pd.DataFrame:
    """
    Loads a parquet file and converts it to the standard blob schema.

    BubbleSAM parquet files store 'area' and 'bbox'. This function computes the
    'center' and 'radius' to make it compatible with downstream analysis.
    OpenCV parquet files already contain these values and so they do not need
    to be computed, in which case the parquet file is loaded and the ``center``
    values are converted from a tuple to separate dataframe columns for x
    and y coordinates.

    Parameters
    ----------
    parquet_path : Path
        The path to the `parquet.gzip` file.
    method : Literal["BubbleSAM", "OpenCV"]
        The detection method that was used to generate the parquet file,
        (i.e. `bubblesam` or `opencv`)

    Returns
    -------
    pd.DataFrame
        A DataFrame with data from performing detection,
        has 'center_x', 'center_y', 'area', 'radius', and 'bbox' columns.
    """
    df = pd.read_parquet(parquet_path)
    if method.lower() == "bubblesam":
        if {"area", "bbox"}.issubset(df.columns):
            # ``bbox`` object is converted to a str before saving parquet
            # in detection module, convert back to list for processing
            bbox_col = df["bbox"].apply(ast.literal_eval)
            bbox_arr = np.array(bbox_col.tolist())
            cy = (bbox_arr[:, 0] + bbox_arr[:, 2]) / 2 
            cx = (bbox_arr[:, 1] + bbox_arr[:, 3]) / 2 
            out = pd.DataFrame({
                "center_x": cx, 
                "center_y": cy,
                "area": df["area"],
                "radius": np.sqrt(df["area"] / np.pi),
                "bbox": bbox_col,
            })
            return out
        else:
            raise ValueError("BubbleSAM file is missing 'area' or 'bbox' columns.")
    # convert the OpenCV center values stored as a tuple into two separate columns
    # TODO: store these values directly during bubble detection for opencv instead
    # of performing programmatically.
    else:
        df[["center_x", "center_y"]] = pd.DataFrame(df["center"].to_list(), index=df.index)
        df.drop(columns=["center"], inplace=True)

    return df

def _drop_invalid_phase_rows(
    df: pd.DataFrame, 
    phase_col: Literal["Phase_Separation"]
) -> pd.DataFrame:
    """
    Removes rows where the specified column denoting phase
    separation status of the data point is NaN or empty. Used
    for instances where a phase status label was not provided
    for any of the per-image data entries that are used to
    generate the aggregated dataset, such that every data point
    to be used for subsequent downstream steps has a ground-truth
    label. 

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to be filtered.
    phase_col : Literal["Phase_Separation"]
        The name of the column storing the phase separation status
        label (0 for single-phase or 1 for two-phase)

    Returns
    -------
    pd.DataFrame
        A filtered DataFrame with invalid rows removed.
    """
    if phase_col not in df.columns:
        return df
    return df.dropna(subset=[phase_col])

def _calculate_nnd_stats(
    points: np.ndarray,
    img_hyp: float,
) -> dict[str, float]:
    """Computes mean and median Nearest-Neighbor Distances (NND) for points.

    Parameters
    ----------
    points : np.ndarray
        An (N, 2) array of (x, y) coordinates. 
    img_hyp : float
        The value of the image hypotenuse for setting the `distance_upper_bound` argument

    Returns
    -------
    dict[str, float]
        A dictionary with 'mean_nnd' and 'median_nnd'.
    """
    # construct the KDTree
    tree = KDTree(points)
    # query the closest two neighbors (the first neighbor is always itself)
    distances, _ = tree.query(points, k=2, distance_upper_bound=img_hyp)
    # ignore the first nearest neighbor (self)
    nnd = distances[:, 1]

    return {"mean_nnd": nnd.mean(), "median_nnd": np.median(nnd)}

def _calculate_voronoi_stats(
    points: np.ndarray
) -> dict[str, float]:
    """
    Computes statistics from the areas of finite Voronoi cells.

    This function calculates the mean, median, and standard deviation of
    the areas of Voronoi cells that are fully contained within the point set.

    Parameters
    ----------
    points : np.ndarray
        An (N, 2) array of (x, y) coordinates. Requires at least 4 points
        for a stable Voronoi tessellation.

    Returns
    -------
    dict[str, float]
        A dictionary containing 'mean_voronoi_area', 'median_voronoi_area',
        'std_voronoi_area'. If no finite areas are found, an empty dictionary
        is returned.
    """
    vor = Voronoi(points) 

    # iterate through regions, filtering out edge regions
    finite_areas = []
    for region_id in vor.point_region:
        verts = vor.regions[region_id]
        # for all valid regions calculate the area of the region
        # using the shoelace formula, filtering out infinite area
        # regions.
        if all(v >= 0 for v in verts):
            poly = vor.vertices[verts]
            x, y = poly[:, 0], poly[:, 1]
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) -
                                np.dot(y, np.roll(x, 1)))
            finite_areas.append(area)
    
    if not finite_areas:
        log.warning("No finite areas found in Voronoi regions.")
        return {} 

    # calculate statistics from finite areas
    areas_arr = np.asarray(finite_areas)
    mean_area = areas_arr.mean()
    std_area = areas_arr.std(ddof=1)
    stats = {
        "mean_voronoi_area": mean_area,
        "median_voronoi_area": np.median(areas_arr),
        "std_voronoi_area": std_area,
    }
    return stats

def _calculate_graph_metrics(
    points: np.ndarray,
    areas: np.ndarray,
    *,
    method: Literal["delaunay", "knn", "radius"],
    r_param: Optional[Union[int, float]] = None,
    k_param: Optional[int] = None,
    img_hyp: float,
) -> dict[str, Any]:
    """Builds a spatial graph from points and calculates network metrics.

    Constructs a graph using Delaunay triangulation, radius search, or k-nearest
    neighbors, then computes metrics like degree, clustering, and component sizes.

    Parameters
    ----------
    points : np.ndarray
        An (N, 2) array of node coordinates.
    areas : np.ndarray
        An (N,) array of detected blob areas, used for Largest Connected
        Component (LCC) area statistics.
    method : Literal["delaunay", "knn", "radius"]
        The graph construction method: 'delaunay', 'radius', or 'knn'.
        Descriptions of each method and their relative merits are provided
        below:

        - ``delaunay``: the set of nodes and edges is defined by the Delaunay
                        triangulation of the input points, i.e. the circumcircle of the
                        nodes forming each triangle contains no other node inside.
                        Generates comprehensive graph of node connectivity and is
                        useful for when the user wants to give more weight to the number
                        of graph edges, for example when bubbles have a more spread out
                        morphology.

        - ``knn``: maps the connections between the k nearest neighboring nodes
                   regardless of the density of the nodes. Will connect sparse
                   nodes but also does not strictly account for all connections within
                   groups of dense nodes, depending on k. The input parameter `k` is
                   modified if the maximum number of neighboring nodes is less than
                   k such that the value of k becomes the number of nodes minus 1.
                   The knn method is useful when trying to characterizing a variety of
                   different compositions with varied morphologies because it provides
                   a generalized view of local and global spatial organization.

        - ``radius``: finds all the connections between nodes within the radius
                      parameter r, which requires user estimation of the relative distance
                      between nodes. Depending on the radius parameter, the graph
                      will be smaller in sparse areas and larger in dense areas. Can
                      also result in no connections being made in graphs where
                      the distance between any two nodes is greater than the radius
                      parameter. The radius method is useful for characterizing compositions
                      with more dense morphologies, where the user wants to emphasize
                      the relationship between closely connected groups of points.
    r_param : Optional[Union[int, float]]
        The radius (in pixels) for 'radius' graphs.
    k_param : Optional[int]
        The k value for 'knn' graphs. k is the maximum number of nearest
        neighbors to use when building the graph. k is overridden when it
        exceeds the number of nodes for a given input to avoid empty dict
        when n_nodes < k.
    img_hyp : float
        The value of the image hypotenuse for setting the `distance_upper_bound` argument
        when performing KDTree query with the `knn` graph method

    Returns
    -------
    dict[str, Any]
        A dictionary of graph metrics. Defaults to 0 for integer values and NaN for
        floating point statistical measures if calculation is not possible.
    """
    metrics = {}
    metrics["graph_num_nodes"] = points.shape[0]

    # check if input method is valid
    if method not in ["knn", "delaunay", "radius"]:
        raise ValueError(
            f"Invalid input parameter for `method`: {method}"
        )
    
    # check that the input parameters for the graph are acceptable for each method
    if method == "knn":
        if not isinstance(k_param, int):
            raise ValueError("`k_param` must be an integer value")
        elif not k_param > 0:
            raise ValueError("`k_param` must be a positive, non-zero integer")
    elif method == "radius":
        if not isinstance(r_param, (int, float)):
            raise ValueError("`r_param` must be either an integer or floating point value")
        elif not r_param > 0:
            raise ValueError("`r_param` must be a positive, non-zero value")

    # initialize `networkx` graph, and add nodes from input data
    n_nodes = points.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from((i, {"area": areas[i], "pos": points[i]})
                         for i in range(n_nodes))
    
    # calculate graph for any of the three input methods
    # calculate the graph using the KDTree to find the
    # closest points within radius set by `r_param`
    if method == "radius":
        tree = KDTree(points)
        # gather point pairs from tree with radius param
        pairs = tree.query_pairs(r=r_param, output_type='ndarray')
        # if pairs exist, find difference between pairs of points
        if len(pairs) != 0:
            diff = np.diff(points[pairs], axis=1)
            # calculate the euclidean distance between pairs of points 
            dist = np.linalg.norm(diff, axis=2)
            # add point, distance pairs to graph edges
            graph.add_edges_from((i, j, {"distance": d}) for (i, j), d in zip(pairs, dist))
    # alternatively calculate the graph using the KDTree
    # to find the k-nearest neighbors of the points as determined
    # by the input `k_param`
    elif method == "knn":
        # k is the minimum of the input parameter
        # and the maximal number of neighbors (nodes-1) 
        # to avoid empty dict entries when n_nodes < k
        k = min(k_param, n_nodes - 1)
        tree = KDTree(points)
        # gather point pairs from tree with knn 
        # index k by 1 because closest node is always itself
        dists, idxs = tree.query(points, k=k + 1, distance_upper_bound=img_hyp)
        # broadcast the first column (node indices) to the shape of the
        # knn array so that we can group the nodes with each of their
        # nearest neighbors
        node_idx = np.broadcast_to(idxs[:, [0]], idxs[:, 1:].shape)
        # group the node, neighbor pairs then stack the sorted
        # pairs with their respective distances
        pairs_idx = np.column_stack((node_idx.ravel(), idxs[:, 1:].ravel()))
        pairs_dists = np.column_stack((pairs_idx, dists[:, 1:].ravel()))
        # add all the edges to the graph
        new_edges = (
            (
                pairs_dists[n, 0],
                pairs_dists[n, 1],
                {"distance": pairs_dists[n, 2]}
            ) for n in range(len(pairs_dists))
        ) 
        graph.add_edges_from(new_edges)

    # alternatively calculate the graph using Delaunay triangulation
    elif method == "delaunay" and n_nodes >= 3:
        # calculate the Delaunay triangulation of input points
        tri = Delaunay(points)
        # get the indices of the points forming the triangles
        tri_sim = tri.simplices
        # shift all points so that we can calculate
        # the distance between adjacent points
        tri_sim_shift = np.roll(tri_sim, shift=-1, axis=1)
        # stack points with neighbors and calculate the
        # euclidean distance between the points
        point_pairs = np.column_stack((tri_sim.ravel(), tri_sim_shift.ravel()))
        diff = np.diff(points[point_pairs], axis=1)
        dist = np.linalg.norm(diff, axis=2)
        # add new edges to the graph
        new_edges = (
            (
                point_pairs[n, 0],
                point_pairs[n, 1],
                {"distance": dist[n]}
            ) for n in range(len(point_pairs))
        )
        graph.add_edges_from(new_edges)
    
    # index graph-based features
    metrics["graph_num_nodes"] = n_nodes
    metrics["graph_num_edges"] = graph.number_of_edges()

    degrees = np.fromiter((d for _, d in graph.degree()), dtype=float)
    metrics["graph_avg_degree"] = degrees.mean()
    metrics["graph_degree_std"] = degrees.std(ddof=1)
    
    metrics["graph_avg_clustering"] = nx.average_clustering(graph)
    nbr_dists = np.fromiter((d["distance"] for _, _, d in
                             graph.edges(data=True)), dtype=float)
    metrics["graph_avg_neighbor_distance"] = nbr_dists.mean()

    components = list(nx.connected_components(graph))
    metrics["graph_num_components"] = len(components)
    lcc = max(components, key=len)
    metrics["graph_lcc_node_fraction"] = len(lcc) / n_nodes
    lcc_areas = [graph.nodes[n]["area"] for n in lcc]
    metrics["graph_avg_node_area_lcc"] = np.mean(lcc_areas)

    return metrics

def _extract_blob_properties(
    df: pd.DataFrame,
    *,
    center_cols: list[str],
    area_col: Literal["area"],
    radius_col: Literal["radius"],
    bbox_col: Literal["bbox"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    """Extracts geometric properties and image size from a blob DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame loaded from a blob data parquet file.
    center_cols: list[str]
        Column names for blob centroids [x, y].
    area_col : Literal["area"]
        Column name for blob areas
    radius_col : Literl["radius"]
        Column name for blob radii.
    bbox_col : Literal["bbox"]
        Column name for bounding boxes (x, y, w, h).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]
        A tuple containing: (centroids, areas, radii, (image_width, image_height)).
        Returns empty arrays/NaN values if data is missing.
    """
    required_cols = {*center_cols, area_col, radius_col, bbox_col}
    if not required_cols.issubset(df.columns) or df.empty:
        return np.array([]), np.array([]), np.array([]), (np.nan, np.nan)

    centroids = df[center_cols].to_numpy()
    areas = df[area_col].to_numpy()
    radii = df[radius_col].to_numpy()

    bbox_array = np.asarray(df[bbox_col].tolist())
    img_w = np.max(bbox_array[:, 2])
    img_h = np.max(bbox_array[:, 3])

    return centroids, areas, radii, (img_w, img_h)

def _calculate_all_spatial_metrics(
    df_blobs: pd.DataFrame,
    *,
    graph_method: Literal["delaunay", "radius", "knn"],
    k_param: Optional[int] = None,
    r_param: Optional[Union[int, float]] = None,
) -> dict[str, Any]:
    """Runs the end-to-end spatial metric calculation for a single image.

    This function serves as a wrapper to extract blob properties and compute
    all blob, coverage, NND, Voronoi, and graph-based metrics.

    Parameters
    ----------
    df_blobs : pd.DataFrame
        The per-blob data table for a single image.
    graph_method : Literal["delaunay", "radius", "knn"]
        The graph construction method ('delaunay', 'radius', or 'knn').
    k_param : Optional[int]
        The k value for graph construction when method == "knn".
    r_param : Optional[Union[int, float]]
        The radius value for graph construction when method == "radius".

    Returns
    -------
    dict[str, Any]
        A dictionary containing all calculated metrics for the image.
    """
    metrics = {
        "num_blobs": 0,
        "mean_blob_area": np.nan,
        "median_blob_area": np.nan,
        "std_blob_area": np.nan,
        "total_blob_area": 0.0,
        "mean_blob_radius": np.nan,
        "median_blob_radius": np.nan,
        "graph_num_nodes": 0,
        "graph_num_edges": 0,
        "graph_avg_degree": np.nan,
        "graph_degree_std": np.nan,
        "graph_num_components": np.nan,
        "graph_lcc_node_fraction": np.nan,
        "graph_avg_clustering": np.nan,
        "graph_avg_neighbor_distance": np.nan,
        "graph_avg_node_area_lcc": np.nan,
        "mean_nnd": np.nan,
        "median_nnd": np.nan,
        "mean_voronoi_area": np.nan,
        "median_voronoi_area": np.nan,
        "std_voronoi_area": np.nan,
    }
    centroids, areas, radii, (w, h) = _extract_blob_properties(
        df_blobs,
        center_cols=["center_x", "center_y"],
        area_col="area",
        radius_col="radius",
        bbox_col="bbox",
    )
    img_area = w * h
    img_hyp = np.hypot(h, w)
    
    metrics.update(
        num_blobs = areas.size,
        mean_blob_area = areas.mean(),
        median_blob_area = np.median(areas),
        std_blob_area = areas.std(ddof=1),
        total_blob_area = areas.sum(),
        mean_blob_radius = radii.mean(),
        median_blob_radius = np.median(radii),
    )

    tba = metrics["total_blob_area"]
    coverage = 100.0 * tba / img_area
    metrics["coverage_percentage"] = coverage
    
    if len(centroids) >= 2:
        metrics.update(_calculate_nnd_stats(centroids, img_hyp))
        metrics.update(
            _calculate_graph_metrics(
                centroids,
                areas,
                method=graph_method,
                k_param=k_param,
                r_param=r_param,
                img_hyp=img_hyp,
            )
        )
        if len(centroids) >= 4:
            metrics.update(_calculate_voronoi_stats(centroids))
    
    return metrics

def _calculate_summary_statistics(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    carry_over_cols: Sequence[str],
    *,
    exclude_numeric_cols: Sequence[str] | None = None,
    exclude_numeric_regex: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Aggregate numeric metrics per group, excluding selected numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Table of per-image metrics.
    group_cols : Sequence[str]
        Columns to group by; must exist in df.
    carry_over_cols : Sequence[str]
        Columns to preserve per group using 'first'.
    exclude_numeric_cols : Sequence[str] | None
        Exact numeric column names to exclude (e.g., ['Offset']).
    exclude_numeric_regex : Sequence[str] | None
        Regex patterns; numeric columns matching any are excluded.

    Returns
    -------
    pd.DataFrame
        One row per group with min/max/median/std for allowed numeric cols.
        Carry-over columns are included without aggregation.
    """
    # check that any of the grouping columns exist in df
    valid_cols = pd.Index(group_cols).intersection(df.columns, sort=False).to_list()
    if not valid_cols:
        raise ValueError(f"None of the grouping columns {group_cols} exist.")

    # ignore unwanted columns based on user input
    df_out = df.loc[
        :, (~df.columns.isin([*(exclude_numeric_cols
            if exclude_numeric_cols else []), *group_cols]))
        & (~df.columns.str.contains("|".join(exclude_numeric_regex or ["$^"])))
    ]

    # collect all remaining columns that contain numerical data
    df_num = df_out.select_dtypes(include="number")

    # find the carry over columns in ``df``
    carry = list(set(carry_over_cols).intersection(df.columns))

    # initialize dictionary keys for aggregation
    _std_agg.__name__ = "std"
    agg_spec = {c: ["min", "max", "median", _std_agg] for c in df_num.columns}
    agg_spec.update({c: ["first"] for c in df[carry].columns})

    grouped = df.groupby(
        valid_cols, as_index=False
    ).agg(agg_spec)  # type: ignore[arg-type]

    grouped.columns = ['_'.join(filter(None, map(str, col))) for col in grouped.columns]
    grouped.rename(columns={f"{c}_first": c for c in df[carry].columns}, inplace=True)

    return grouped

def _process_parquet_files(
    input_dir: Path,
    *,
    mode: Literal["OpenCV", "BubbleSAM"],
    graph_method: Literal["delaunay", "radius", "knn"],
    k_param: int | None = None,
    r_param: int | float | None = None,
    time_label: str | None = None,
) -> pd.DataFrame:
    """
    Scans a directory, computes spatial metrics for 
    each file, and returns a DataFrame.

    This function recursively searches for parquet files, 
    parses metadata from their filenames, loads them, 
    and computes a suite of spatial metrics.

    Parameters
    ----------
    input_dir : Path
        The root directory to search for parquet files.
    mode : Literal["OpenCV", "BubbleSAM"]
        Processing mode, 'OpenCV' or 'BubbleSAM', which determines
        which files to look for and how to parse them.
    graph_method : Literal["delaunay", "radius", "knn"]
        The graph construction method to use ('delaunay', 'radius', 'knn').
    k_param : Optional[int]
        Parameter for ``knn`` graph construction.
    r_param : Optional[int | float]
        Parameter for the ``radius`` graph construction.
    time_label : Optional[str]
        A label to assign to the 'Time' column for all processed files.
        `Time` denotes the collection period for the data point, either
        immediately after mixing (1st) or 4 hours after mixing (2nd)

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row contains the metrics for one processed image.
    """
    rows = []

    if mode.lower() == "opencv":
        glob_pattern = "*_bubble_data.parquet.gzip"
    elif mode.lower() == "bubblesam":
        glob_pattern = "*_masks_filtered.parquet.gzip"
    else:
        raise ValueError("Mode must be either 'OpenCV' or 'BubbleSAM'.")

    all_parquets = list(input_dir.rglob(glob_pattern)) 
    for parquet_path in tqdm(
        all_parquets,
        total=len(all_parquets),
        desc="Processing Parquet Files"
    ):
        metadata = _parse_filename(parquet_path.name, mode)
        if not metadata:
            warnings.warn(f"Could not parse metadata from filename: {parquet_path.name}")
            continue
        if time_label:
            metadata["Time"] = time_label
        try:
            df_blobs = _load_df(parquet_path, mode)
        except (ValueError, ArrowInvalid) as exc:
            warnings.warn(
                f"Failed to load or parse {parquet_path}: {type(exc).__name__}({exc})"
            )
            continue
        metrics = _calculate_all_spatial_metrics(
            df_blobs, graph_method=graph_method, k_param=k_param, r_param=r_param,
        )
        metrics["image_name"] = parquet_path.name.replace("parquet.gzip", "tiff")
        metrics.update(metadata)
        rows.append(metrics)

    if not rows:
        raise FileNotFoundError(
            f"No valid files were processed in {input_dir} "
            f"for mode '{mode}'.")
    return pd.DataFrame(rows)

def full_analysis(
    *,
    input_dir: Path,
    per_image_csv: Path,
    aggregate_csv: Path,
    mode: Literal["OpenCV", "BubbleSAM"],
    graph_method: Literal["delaunay", "radius", "knn"],
    r_param: int | float | None = None,
    k_param: int | None = None,
    composition_csv: Path | None = None,
    cols_to_add: Sequence[str] | None = None,
    group_cols: Sequence[str] | None = None,
    carry_over_cols: Sequence[str] | None = None,
    time_label: str | None = None,
    exclude_numeric_cols: list[str] | None = None,
    exclude_numeric_regex: list[str] | None = None,
) -> None:
    """Executes the complete data analysis pipeline.

    This function orchestrates the entire workflow:
    1. Processes a directory of blob data to get per-image metrics.
    2. Saves the per-image metrics to a CSV file.
    3. Merges blob data with user provided composition data including
       values for the weight percentages of each polymer in the
       composition as well as the phase separation ground-truth label
       of the input image.
    4. Aggregates the metrics into summary statistics.
    5. Cleans the final aggregated data and saves it to another CSV.

    Parameters
    ----------
    input_dir : Path
        The root directory containing the raw per-image parquet files.
    per_image_csv : Path
        The path to save the per-image metrics CSV file.
    aggregate_csv : Path
        The path to save the final aggregated metrics CSV file.
    mode : Literal["OpenCV", "BubbleSAM"]
        The processing mode, either 'OpenCV' or 'BubbleSAM'.
    graph_method : Literal["delaunay", "radius", "knn"]
        The graph topology method ('delaunay', 'radius', 'knn').
    r_param : Optional[Union[int, float]]
        The radius (in pixels) for 'radius' graphs.
    k_param : Optional[int]
        The k value for 'knn' graphs. k is the maximum number of nearest
        neighbors to use when building the graph. k is overridden when it
        exceeds the number of nodes for a given input to avoid empty dict
        when n_nodes < k.
    composition_csv : Optional[Path]
        Path to an external composition data table to merge.
    cols_to_add : Optional[Sequence[str]]
        A list of columns from `composition_csv` to add.
    group_cols : Optional[Sequence[str]]
        Columns to group by for aggregation.
    carry_over_cols : Optional[Sequence[str]]
        Non-numeric columns to preserve during aggregation.
    time_label : Optional[str]
        A label to assign to the 'Time' metadata column, denoting the
        collection time-point of the sample data.
    exclude_numeric_cols : list[str] | None
        Exact numeric columns to exclude from aggregation.
    exclude_numeric_regex : list[str] | None
        Regex patterns; matching numeric columns are excluded.
    """
    # iterate through all parquet files and return a dataframe
    # containing per image statistics
    per_img_df = _process_parquet_files(
        input_dir,
        mode=mode,
        graph_method=graph_method,
        k_param=k_param,
        r_param=r_param,
        time_label=time_label,
    )

    # re-order df columns to put the file information first
    id_cols = ["image_name", "Offset", "Position", "Label", "Class", "Time", "UniqueID"]
    id_cols = [c for c in id_cols if c in per_img_df.columns]
    ordered_cols = id_cols + [c for c in per_img_df.columns if c not in id_cols]
    per_img_df = per_img_df[ordered_cols]
    per_image_csv.parent.mkdir(parents=True, exist_ok=True)
    per_img_df.to_csv(per_image_csv, index=False)
    log.info(f"Per-image metrics saved to: {per_image_csv}")

    # merge the per image dataframe with user specified columns
    # from the composition dataframe on ``UniqueID``
    if composition_csv and cols_to_add:
        comp_df = pd.read_csv(composition_csv)
        per_img_df = _merge_composition_data(
            per_img_df, comp_df, cols_to_add=cols_to_add, merge_key="UniqueID"
        )

    # determine the df columns on which to aggregate statistics (user specified
    # or default) and which columns to preserve without aggregating
    final_group_cols = group_cols or ["Group", "Label", "Time", "Class"]
    final_carry_cols = carry_over_cols or []
    # aggregate per image statistics 
    agg_df = _calculate_summary_statistics(
        per_img_df,
        group_cols=final_group_cols,
        carry_over_cols=final_carry_cols,
        exclude_numeric_cols=exclude_numeric_cols,
        exclude_numeric_regex=exclude_numeric_regex,
    )

    # remove rows with NaN or empty values and save aggregated df
    if "Phase_Separation" in agg_df.columns:
        agg_df.dropna(subset=["Phase_Separation"], inplace=True)
    aggregate_csv.parent.mkdir(parents=True, exist_ok=True)
    agg_df.to_csv(aggregate_csv, index=False)
    log.info(f"Aggregated metrics saved to: {aggregate_csv}")
