import re
import warnings
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import KDTree, Voronoi, Delaunay, QhullError
import logging
from tqdm.auto import tqdm

__all__: Sequence[str] = [
    "full_analysis"
]

log = logging.getLogger(__name__)

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
        A list of columns from composition_df to add to summary_df.
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

def _parse_filename(fname: str, method: str) -> dict[str, Any]:
    """
    Parses an OpenCV-generated filename to extract metadata.

    Parameters
    ----------
    fname : str
        The filename, e.g., 'offset -1_center_A1_Bf_Raw_uuid_bubble_data.parquet.gzip'.
    method : str
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
        r"(?:(bottom|top|left|right|center)_)?"
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


def _load_df(parquet_path: Path, method: str) -> pd.DataFrame:
    """
    Loads a parquet file and converts it to the standard blob schema.

    BubbleSAM parquet store 'area' and 'bbox'. This function computes the
    'center' and 'radius' to make it compatible with downstream analysis.

    Parameters
    ----------
    parquet_path : Path
        The path to the `parquet.gzip` file.

    Returns
    -------
    pd.DataFrame
        A DataFrame with data from performing detection,
        has 'center', 'area', 'radius', and 'bbox' columns
        when method == bubblesam.
    """
    df = pd.read_parquet(parquet_path)
    if method.lower() == "bubblesam":
        if {"area", "bbox"}.issubset(df.columns):
            bbox_list = df["bbox"].tolist()
            cy = [(b[0] + b[2]) / 2 for b in bbox_list]
            cx = [(b[1] + b[3]) / 2 for b in bbox_list]
            out = pd.DataFrame({
                "center": list(zip(cx, cy)),
                "area":   df["area"].astype(float),
                "radius": np.sqrt(df["area"].astype(float) / np.pi),
                "bbox":   df["bbox"],
            })
            return out
        else:
            raise ValueError("BubbleSAM file is missing 'area' or 'bbox' columns.")
    return df

def _drop_invalid_phase_rows(
    df: pd.DataFrame, 
    phase_col: str
) -> pd.DataFrame:
    """
    Removes rows where the specified phase column is NaN or empty.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to be filtered.
    phase_col : str
        The name of the column to check for valid phase data.

    Returns
    -------
    pd.DataFrame
        A filtered DataFrame with invalid rows removed.
    """
    if phase_col not in df.columns:
        return df
    df[phase_col] = df[phase_col].replace('', np.nan)
    return df.dropna(subset=[phase_col])

def _calculate_nnd_stats(
    points: Optional[np.ndarray]
) -> dict[str, float]:
    """Computes mean and median Nearest-Neighbor Distances (NND) for points.

    Parameters
    ----------
    points : Optional[np.ndarray]
        An (N, 2) array of (x, y) coordinates. Returns NaNs if input is
        None or has fewer than two points.

    Returns
    -------
    dict[str, float]
        A dictionary with 'mean_nnd' and 'median_nnd'. Values are NaN
        if the calculation is not possible.
    """
    if points is None or points.shape[0] < 2:
        return {"mean_nnd": np.nan, "median_nnd": np.nan}

    tree = KDTree(points)
    distances, _ = tree.query(points, k=2)
    nnd = distances[:, 1]
    nnd = nnd[np.isfinite(nnd)]
    if nnd.size == 0:
        warnings.warn("NND calculation failed: No finite neighbor distances found.")
        return {"mean_nnd": np.nan, "median_nnd": np.nan}

    return {"mean_nnd": nnd.mean(), "median_nnd": np.median(nnd)}

def _calculate_voronoi_stats(
    points: Optional[np.ndarray]
) -> dict[str, float]:
    """
    Computes statistics from the areas of finite Voronoi cells.

    This function calculates the mean, median, standard deviation, and
    coefficient of variation for the areas of Voronoi cells that are
    fully contained within the point set.

    Parameters
    ----------
    points : Optional[np.ndarray]
        An (N, 2) array of (x, y) coordinates. Requires at least 4 points
        for a stable Voronoi tessellation.

    Returns
    -------
    dict[str, float]
        A dictionary containing 'mean_voronoi_area', 'median_voronoi_area',
        'std_voronoi_area', and 'cv_voronoi_area'. Values are NaN on failure.
    """
    stats = {
        "mean_voronoi_area": np.nan,
        "median_voronoi_area": np.nan,
        "std_voronoi_area": np.nan,
        "cv_voronoi_area": np.nan,
    }
    if points is None or points.shape[0] < 4:
        return stats

    try:
        # add jitter to input data to prevent voronoi crashing
        # due to evenly spaced data points
        jitter_scale = 1e-6 * np.ptp(points, axis=0).mean()
        vor = Voronoi(points + np.random.normal(0.0, jitter_scale, points.shape))
    
        # iterate through regions, filtering out edge regions
        finite_areas = []
        for region_id in vor.point_region:
            verts = vor.regions[region_id]
            # for all valid regions calculate the area of the region
            # using the shoelace formula, filtering out infinite area
            # regions and extremely small regions.
            if verts and all(v >= 0 for v in verts):
                poly = vor.vertices[verts]
                if poly.shape[0] >= 3:
                    x, y = poly[:, 0], poly[:, 1]
                    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) -
                                        np.dot(y, np.roll(x, 1)))
                    if np.isfinite(area) and area > 1e-9:
                        finite_areas.append(area)

        if not finite_areas:
            return stats
        # calculate statistics from finite areas
        # `cv` -> coefficient of variation
        areas_arr = np.asarray(finite_areas)
        mean_area = areas_arr.mean()
        std_area = areas_arr.std(ddof=0)
        stats.update(
            mean_voronoi_area = mean_area,
            median_voronoi_area = np.median(areas_arr),
            std_voronoi_area = std_area,
            cv_voronoi_area = (std_area / mean_area) if mean_area else np.nan,
        )
        return stats
    except (QhullError, ValueError) as exc:
        warnings.warn(f"Voronoi calculation failed: {exc}")
        return stats

def _calculate_graph_metrics(
    points: np.ndarray,
    areas: np.ndarray,
    *,
    method: str,
    param: Optional[Union[int, float]] = None,
) -> dict[str, Any]:
    """Builds a spatial graph from points and calculates network metrics.

    Constructs a graph using Delaunay triangulation, radius search, or k-nearest
    neighbors, then computes metrics like degree, clustering, and component sizes.

    Parameters
    ----------
    points : np.ndarray
        An (N, 2) array of node coordinates.
    areas : np.ndarray
        An (N,) array of node areas, used for Largest Connected
        Component (LCC) area statistics.
    method : str
        The graph construction method: 'delaunay', 'radius', or 'knn'.
    param : Optional[Union[int, float]]
        The radius for 'radius' graphs or k for 'knn' graphs.

    Returns
    -------
    dict[str, Any]
        A dictionary of graph metrics. Defaults to 0 for counts and NaN for
        statistical measures if calculation is not possible.
    """
    # initialize empty dictionary to fill or return if insufficient data points
    baseline = {
        "graph_num_nodes": 0,
        "graph_num_edges": 0,
        "graph_avg_degree": np.nan,
        "graph_degree_std": np.nan,
        "graph_num_components": np.nan,
        "graph_lcc_node_fraction": np.nan,
        "graph_avg_clustering": np.nan,
        "graph_avg_neighbor_distance": np.nan,
        "graph_avg_node_area_lcc": np.nan,
    }

    # check if input is empty or insufficient datapoints, if so return baseline
    if (points is None or areas is None or points.shape[0] < 2 or
            points.shape[0] != areas.shape[0]):
        baseline["graph_num_nodes"] = 0 if points is None else int(points.shape[0])
        return baseline
    # check if input method is valid
    if method not in ["knn", "delaunay", "radius"]:
        raise ValueError(
            f"Invalid input parameters for `method`: {method}"
        )
    
    # initialize `networkx` graph, and add nodes from input data
    n_nodes = points.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from((i, {"area": areas[i], "pos": points[i]})
                         for i in range(n_nodes))
    
    # calculate graph for any of the three input methods
    try:
        # calculate the graph using the KDTree to find the
        # closest points within radius set by `param`
        # https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.spatial.KDTree.html
        if method == "radius" and isinstance(param, (int, float)) and param > 0:
            tree = KDTree(points)
            # gather point pairs from tree with radius param
            pairs = list(tree.query_pairs(r=param))
            # find difference between pairs of points
            diff = np.diff(points[pairs], axis=1)
            # calculate the euclidian distance between pairs of points 
            dist = np.linalg.norm(diff, axis=2)
            # add point, distance pairs to graph edges
            graph.add_edges_from((i, j, {"distance": d}) for (i, j), d in zip(pairs, dist))
        # alternatively calculate the graph using the KDTree
        # to find the k-nearest neighbors of the points as determined
        # by the input `param`
        elif method == "knn" and isinstance(param, int) and param > 0:
            # k is the minimum of the input parameter
            # OR the maximal number of neighbors (nodes-1) 
            k = min(param, n_nodes - 1)
            tree = KDTree(points)
            # gather point pairs from tree with knn 
            dists, idxs = tree.query(points, k=k + 1)
            # broadcast the first column (node indices) to the shape of the
            # knn array so that we can group the nodes with each of their
            # nearest neighbors
            node_idx = np.broadcast_to(idxs[:, [0]], idxs[:, 1:].shape)
            # group the node, neighbor pairs and sort them by value
            # then stack the sorted pairs with their respective distances
            # and find the unique node-edge pairs in the array
            pairs_idx = np.column_stack((node_idx.ravel(), idxs[:, 1:].ravel()))
            sorted_pairs = np.sort(pairs_idx, axis=1)
            pairs_dists = np.column_stack((sorted_pairs, dists[:, 1:].ravel()))
            unique_pairs = np.unique(pairs_dists, axis=0)
            # add all the edges to the graph
            new_edges = (
                (
                    unique_pairs[n, 0],
                    unique_pairs[n, 1],
                    {"distance": unique_pairs[n, 2]}
                ) for n in range(len(unique_pairs))
            ) 
            graph.add_edges_from(new_edges)

        # alternatively calculate the graph using Delaunay triangulation
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html
        elif method == "delaunay" and n_nodes >= 3:
            # calculate the delaunay triangulation of input points
            tri = Delaunay(points)
            # get the indices of the points forming the triangles
            tri_sim = tri.simplices
            # shift all points so that we can calculate
            # the distance between adjacent points
            tri_sim_shift = np.roll(tri_sim, shift=-1, axis=1)
            # stack points and neighbors, remove non-unique pairs
            # and calculate the euclidean distance between the points
            point_pairs = np.column_stack((tri_sim.ravel(), tri_sim_shift.ravel()))
            unique_pairs = np.unique(np.sort(point_pairs, axis=1), axis=0)
            diff = np.diff(points[unique_pairs], axis=1)
            dist = np.linalg.norm(diff, axis=2)
            # add new edges to the graph
            new_edges = (
                (
                    unique_pairs[n, 0],
                    unique_pairs[n, 1],
                    {"distance": dist[n]}
                ) for n in range(len(unique_pairs))
            )
            graph.add_edges_from(new_edges)
    except Exception as exc:
        warnings.warn(f"Graph construction ({method}) failed: {exc}")
    
    # index graph-based features
    metrics = baseline.copy()
    metrics["graph_num_nodes"] = n_nodes
    metrics["graph_num_edges"] = graph.number_of_edges()

    degrees = np.fromiter((d for _, d in graph.degree()), dtype=float)
    if degrees.size > 0:
        metrics["graph_avg_degree"] = degrees.mean()
        metrics["graph_degree_std"] = degrees.std(ddof=0)

    if graph.number_of_edges() > 0:
        metrics["graph_avg_clustering"] = nx.average_clustering(graph)
        nbr_dists = np.fromiter((d["distance"] for _, _, d in
                                 graph.edges(data=True)), dtype=float)
        metrics["graph_avg_neighbor_distance"] = nbr_dists.mean()

    components = list(nx.connected_components(graph))
    metrics["graph_num_components"] = len(components)
    if components:
        lcc = max(components, key=len)
        metrics["graph_lcc_node_fraction"] = len(lcc) / n_nodes
        lcc_areas = [graph.nodes[n]["area"] for n in lcc
                     if np.isfinite(graph.nodes[n]["area"])]
        if lcc_areas:
            metrics["graph_avg_node_area_lcc"] = np.mean(lcc_areas)

    return metrics

def _extract_blob_properties(
    df: pd.DataFrame,
    *,
    center_col: str,
    area_col: str,
    radius_col: str,
    bbox_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    """Extracts geometric properties and image size from a blob DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame loaded from a blob data parquet file.
    center_col : str
        Column name for blob centroids (x, y).
    area_col : str
        Column name for blob areas
    radius_col : str
        Column name for blob radii.
    bbox_col : str
        Column name for bounding boxes (x, y, w, h).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]
        A tuple containing: (centroids, areas, radii, (image_width, image_height)).
        Returns empty arrays/NaN values if data is missing.
    """
    required_cols = {center_col, area_col, radius_col, bbox_col}
    if not required_cols.issubset(df.columns) or df.empty:
        return np.array([]), np.array([]), np.array([]), (np.nan, np.nan)

    centroids = np.asarray(df[center_col].tolist(), dtype=float)
    areas = df[area_col].to_numpy(dtype=float)
    radii = df[radius_col].to_numpy(dtype=float)

    bbox_array = np.asarray(df[bbox_col].tolist(), dtype=float)
    img_w = np.nanmax(bbox_array[:, 2]) if bbox_array.size else np.nan
    img_h = np.nanmax(bbox_array[:, 3]) if bbox_array.size else np.nan

    return centroids, areas, radii, (img_w, img_h)

def _calculate_all_spatial_metrics(
    df_blobs: pd.DataFrame,
    *,
    graph_method: str,
    graph_param: Optional[Union[int, float]] = None,
) -> dict[str, Any]:
    """Runs the end-to-end spatial metric calculation for a single image.

    This function serves as a wrapper to extract blob properties and compute
    all blob, coverage, NND, Voronoi, and graph-based metrics.

    Parameters
    ----------
    df_blobs : pd.DataFrame
        The per-blob data table for a single image.
    graph_method : str
        The graph construction method ('delaunay', 'radius', or 'knn').
    graph_param : Optional[Union[int, float]]
        The radius or k value for graph construction.

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
    }
    centroids, areas, radii, (w, h) = _extract_blob_properties(
        df_blobs,
        center_col="center",
        area_col="area",
        radius_col="radius",
        bbox_col="bbox",
    )
    img_area = w * h if np.isfinite(w) and np.isfinite(h) else np.nan

    if areas is not None and radii is not None and areas.size > 0:
        metrics.update(
            num_blobs = areas.size,
            mean_blob_area = areas.mean(),
            median_blob_area = np.median(areas),
            std_blob_area = areas.std(ddof=0),
            total_blob_area = areas.sum(),
            mean_blob_radius = radii.mean(),
            median_blob_radius = np.median(radii),
        )

    tba = metrics["total_blob_area"]
    coverage = (100.0 * tba / img_area) if np.isfinite(img_area) and img_area > 0 else np.nan
    metrics["coverage_percentage"] = coverage

    metrics.update(_calculate_nnd_stats(centroids))
    metrics.update(_calculate_voronoi_stats(centroids))
    metrics.update(
        _calculate_graph_metrics(
            centroids, areas, method=graph_method, param=graph_param
        )
    )
    return metrics

def _calculate_summary_statistics(
    df: pd.DataFrame,
    group_cols: list[str],
    carry_over_cols: list[str],
    *,
    exclude_numeric_cols: list[str] | None = None,
    exclude_numeric_regex: list[str] | None = None,
) -> pd.DataFrame:
    """
    Aggregate numeric metrics per group, excluding selected numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Table of per-image metrics.
    group_cols : list[str]
        Columns to group by; must exist in df.
    carry_over_cols : list[str]
        Columns to preserve per group using 'first'.
    exclude_numeric_cols : list[str] | None
        Exact numeric column names to exclude (e.g., ['Offset']).
    exclude_numeric_regex : list[str] | None
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
        :, (~df.columns.isin((exclude_numeric_cols
            if exclude_numeric_cols else []) + group_cols))
        & (~df.columns.str.contains("|".join(exclude_numeric_regex or ["$^"])))
    ]

    # collect all remaining columns that contain numerical data
    df_num = df_out.select_dtypes(include="number")

    # if no numerical columns exist, return a dataframe with just the
    # user provided columns
    carry = list(set(carry_over_cols).intersection(df.columns))
    if df_num.empty:
        return df[group_cols + carry].drop_duplicates().reset_index(drop=True)

    # initialize dictionary keys for aggregation
    agg_spec = {c: ["min", "max", "median", "std"] for c in df_num.columns}
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
    mode: str,
    graph_method: str,
    graph_param: int | float | None = None,
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
    mode : str
        Processing mode, 'OpenCV' or 'BubbleSAM', which determines
        which files to look for and how to parse them.
    graph_method : str
        The graph construction method to use ('delaunay', 'radius', 'knn').
    graph_param : Optional[int | float]
        Parameter for the graph construction (radius or k).
    time_label : Optional[str]
        A label to assign to the 'Time' column for all processed files.

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
        except Exception as exc:
            warnings.warn(
                f"Failed to load or parse {parquet_path}: {type(exc).__name__}({exc})"
            )
            continue
        metrics = _calculate_all_spatial_metrics(
            df_blobs, graph_method=graph_method, graph_param=graph_param
        )
        metrics["image_name"] =  parquet_path.name.replace("parquet.gzip", "tiff")
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
    mode: str,
    graph_method: str,
    graph_param: int | float | None = None,
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
    3. Merges with external composition data, if provided.
    4. Aggregates the metrics into summary statistics.
    5. Cleans the final aggregated data and saves it to another CSV.

    Parameters
    ----------
    input_dir : Path
        The root directory containing the raw per-blob parquet files.
    per_image_csv : Path
        The path to save the per-image metrics CSV file.
    aggregate_csv : Path
        The path to save the final aggregated metrics CSV file.
    mode : str
        The processing mode, either 'OpenCV' or 'BubbleSAM'.
    graph_method : str
        The graph topology method ('delaunay', 'radius', 'knn').
    graph_param : Optional[int | float]
        Parameter for the graph construction method.
    composition_csv : Optional[Path]
        Path to an external composition data table to merge.
    cols_to_add : Optional[Sequence[str]]
        A list of columns from `composition_csv` to add.
    group_cols : Optional[Sequence[str]]
        Columns to group by for aggregation.
    carry_over_cols : Optional[Sequence[str]]
        Non-numeric columns to preserve during aggregation.
    time_label : Optional[str]
        A label to assign to the 'Time' metadata column.
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
        graph_param=graph_param,
        time_label=time_label,
    )

    # re-order df columns to put the file information first
    id_cols = ["image_name", "Offset", "Position", "Label", "Class", "Time", "UniqueID"]
    ordered_cols = id_cols + [c for c in per_img_df.columns if c not in id_cols]
    per_img_df = per_img_df[ordered_cols]
    per_image_csv.parent.mkdir(parents=True, exist_ok=True)
    per_img_df.to_csv(per_image_csv, index=False)
    log.info(f"Per-image metrics saved to: {per_image_csv}")

    # merge the per image dataframe with user sepcified columns
    # from the composition dataframe on ``UniqueID``
    if composition_csv and cols_to_add:
        comp_df = pd.read_csv(composition_csv)
        per_img_df = _merge_composition_data(
            per_img_df, comp_df, cols_to_add=cols_to_add, merge_key="UniqueID"
        )

    # determine which df columns on which to aggregate statistics (user specified
    # or default) and which columns to preserve without aggregating
    final_group_cols = list(group_cols or ["Group", "Label", "Time", "Class"])
    final_carry_cols = list(carry_over_cols or [])
    # aggregate per image statistics 
    agg_df = _calculate_summary_statistics(
        per_img_df,
        group_cols=final_group_cols,
        carry_over_cols=final_carry_cols,
        exclude_numeric_cols=exclude_numeric_cols,
        exclude_numeric_regex=exclude_numeric_regex,
    )

    # remove rows with NaN or empty values and save aggregated df
    agg_clean_df = _drop_invalid_phase_rows(agg_df, 'Phase_Separation')
    aggregate_csv.parent.mkdir(parents=True, exist_ok=True)
    agg_clean_df.to_csv(aggregate_csv, index=False)
    log.info(f"Aggregated metrics saved to: {aggregate_csv}")
