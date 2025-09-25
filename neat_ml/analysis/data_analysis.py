import re
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import pickle
from scipy.spatial import KDTree, Voronoi, Delaunay, QhullError

__all__: Sequence[str] = [
    "full_analysis"
]

def merge_composition_data(
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

def _parse_opencv_filename(fname: str) -> Dict[str, Any]:
    """
    Parses an OpenCV-generated filename to extract metadata.

    Parameters
    ----------
    fname : str
        The filename, e.g., 'offset -1_center_A1_Bf_Raw_uuid_bubble_data.pkl'.

    Returns
    -------
    Dict[str, Any]
        A dictionary with 'UniqueID', 'Class', 'Offset', 'Position', and 'Label',
        or an empty dictionary if the pattern does not match.
    """
    _OCV_RE = re.compile(
        r"offset\s*(-?\d+)_"
        r"(?:(bottom|top|left|right|center)_)?"
        r"([A-Z]\d+)_"
        r".+?_"
        r"(Bf|Ph)_Raw_"
        r"([0-9a-f\-]+)_bubble_data\.pkl",
        re.IGNORECASE | re.VERBOSE,
    )
    match = _OCV_RE.match(fname)
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

def _load_opencv_df(p: Path) -> pd.DataFrame:
    """Loads a standard OpenCV blob data pickle file.

    Parameters
    ----------
    p : Path
        The path to the `*_bubble_data.pkl` file.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame.
    """
    return pd.read_pickle(p)

def _parse_bubblesam_filename(
    fname: str
) -> Dict[str, Any]:
    """
    Parses a BubbleSAM-generated filename to extract metadata.

    Parameters
    ----------
    fname : str
        The filename, e.g., 'offset -1_center_A1_Bf_Raw_uuid_masks_filtered.pkl'.

    Returns
    -------
    Dict[str, Any]
        A dictionary with metadata, or an empty dictionary on mismatch.
    """
    _BSAM_RE = re.compile(
        r"offset\s*(-?\d+)_"
        r"(bottom|top|left|right|center)_"
        r"([A-Z]\d+)_"
        r".+?_"
        r"(Bf|Ph)_Raw_"
        r"([0-9a-f\-]+)_masks_filtered\.pkl",
        re.IGNORECASE | re.VERBOSE,
    )
    match = _BSAM_RE.match(fname)
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

def _load_bubblesam_df(p: Path) -> pd.DataFrame:
    """
    Loads a BubbleSAM pickle and converts it to the standard blob schema.

    BubbleSAM pickles store 'area' and 'bbox'. This function computes the
    'center' and 'radius' to make it compatible with downstream analysis.

    Parameters
    ----------
    p : Path
        The path to the `*_masks_filtered.pkl` file.

    Returns
    -------
    pd.DataFrame
        A DataFrame with 'center', 'area', 'radius', and 'bbox' columns.
    """
    df = pd.read_pickle(p)
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
    raise ValueError("BubbleSAM file is missing 'area' or 'bbox' columns.")

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
    mask = df[phase_col].notna() & df[phase_col].astype(str).str.strip().ne("")
    return df.loc[mask].reset_index(drop=True)

def calculate_nnd_stats(
    points: Optional[np.ndarray]
) -> Dict[str, float]:
    """Computes mean and median Nearest-Neighbor Distances (NND) for points.

    Parameters
    ----------
    points : Optional[np.ndarray]
        An (N, 2) array of (x, y) coordinates. Returns NaNs if input is
        None or has fewer than two points.

    Returns
    -------
    Dict[str, float]
        A dictionary with 'mean_nnd' and 'median_nnd'. Values are NaN
        if the calculation is not possible.
    """
    if points is None or points.shape[0] < 2:
        return {"mean_nnd": np.nan, "median_nnd": np.nan}

    try:
        tree = KDTree(points)
        distances, _ = tree.query(points, k=2)
        nnd = distances[:, 1]
        nnd = nnd[np.isfinite(nnd)]
        if nnd.size == 0:
            raise ValueError("No finite neighbor distances found.")

        return {"mean_nnd": float(nnd.mean()), "median_nnd": float(np.median(nnd))}
    except ValueError as exc:
        warnings.warn(f"NND calculation failed: {exc}")
        return {"mean_nnd": np.nan, "median_nnd": np.nan}

def calculate_voronoi_stats(
    points: Optional[np.ndarray]
) -> Dict[str, float]:
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
    Dict[str, float]
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
        jitter_scale = 1e-6 * np.ptp(points, axis=0).mean()
        vor = Voronoi(points + np.random.normal(0.0, jitter_scale, points.shape))

        finite_areas: list[float] = []
        for region_id in vor.point_region:
            if region_id == -1:
                continue
            verts = vor.regions[region_id]
            if verts and all(v >= 0 for v in verts):
                poly = vor.vertices[verts]
                if poly.shape[0] >= 3:
                    x, y = poly[:, 0], poly[:, 1]
                    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) -
                                        np.dot(y, np.roll(x, 1)))
                    if np.isfinite(area) and area > 1e-9:
                        finite_areas.append(float(area))

        if not finite_areas:
            return stats

        areas_arr = np.asarray(finite_areas)
        mean_area = areas_arr.mean()
        std_area = areas_arr.std(ddof=0)
        stats.update(
            mean_voronoi_area=float(mean_area),
            median_voronoi_area=float(np.median(areas_arr)),
            std_voronoi_area=float(std_area),
            cv_voronoi_area=float(std_area / mean_area) if mean_area else np.nan,
        )
        return stats
    except (QhullError, ValueError) as exc:
        warnings.warn(f"Voronoi calculation failed: {exc}")
        return stats

def calculate_graph_metrics(
    points: Optional[np.ndarray],
    areas: Optional[np.ndarray],
    *,
    method: str | None,
    param: Optional[Union[int, float]] = None,
) -> Dict[str, Any]:
    """Builds a spatial graph from points and calculates network metrics.

    Constructs a graph using Delaunay triangulation, radius search, or k-nearest
    neighbors, then computes metrics like degree, clustering, and component sizes.

    Parameters
    ----------
    points : Optional[np.ndarray]
        An (N, 2) array of node coordinates.
    areas : Optional[np.ndarray]
        An (N,) array of node areas, used for Largest Connected
        Component (LCC) area statistics.
    method : str
        The graph construction method: 'delaunay', 'radius', or 'knn'.
    param : Optional[Union[int, float]]
        The radius for 'radius' graphs or k for 'knn' graphs.

    Returns
    -------
    Dict[str, Any]
        A dictionary of graph metrics. Defaults to 0 for counts and NaN for
        statistical measures if calculation is not possible.
    """
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

    if (points is None or areas is None or points.shape[0] < 2 or
            points.shape[0] != areas.shape[0]):
        baseline["graph_num_nodes"] = 0 if points is None else int(points.shape[0])
        return baseline

    n_nodes = int(points.shape[0])
    graph = nx.Graph()
    graph.add_nodes_from((i, {"area": float(areas[i]), "pos": points[i]})
                         for i in range(n_nodes))

    try:
        if method == "radius" and isinstance(param, (int, float)) and param > 0:
            tree = KDTree(points)
            for i, j in tree.query_pairs(r=float(param)):
                dist = float(np.linalg.norm(points[i] - points[j]))
                graph.add_edge(i, j, distance=dist)
        elif method == "knn" and isinstance(param, int) and param > 0:
            k = min(param, n_nodes - 1)
            tree = KDTree(points)
            dists, idxs = tree.query(points, k=k + 1)
            for i in range(n_nodes):
                for nn_idx in range(1, k + 1):
                    j = int(idxs[i, nn_idx])
                    if not graph.has_edge(i, j):
                        dist = float(dists[i, nn_idx])
                        graph.add_edge(i, j, distance=dist)
        elif method == "delaunay" and n_nodes >= 3:
            tri = Delaunay(points)
            for simplex in tri.simplices:
                for i, j in zip(simplex, np.roll(simplex, -1)):
                    if not graph.has_edge(i, j):
                        dist = float(np.linalg.norm(points[i] - points[j]))
                        graph.add_edge(int(i), int(j), distance=dist)
    except Exception as exc:
        warnings.warn(f"Graph construction ({method}) failed: {exc}")

    metrics = baseline.copy()
    metrics["graph_num_nodes"] = n_nodes
    metrics["graph_num_edges"] = graph.number_of_edges()

    degrees = np.fromiter((d for _, d in graph.degree()), dtype=float)
    if degrees.size > 0:
        metrics["graph_avg_degree"] = float(degrees.mean())
        metrics["graph_degree_std"] = float(degrees.std(ddof=0))

    if graph.number_of_edges() > 0:
        metrics["graph_avg_clustering"] = float(nx.average_clustering(graph))
        nbr_dists = np.fromiter((d["distance"] for _, _, d in
                                 graph.edges(data=True)), dtype=float)
        metrics["graph_avg_neighbor_distance"] = float(nbr_dists.mean())

    components = list(nx.connected_components(graph))
    metrics["graph_num_components"] = len(components)
    if components:
        lcc = max(components, key=len)
        metrics["graph_lcc_node_fraction"] = len(lcc) / n_nodes
        lcc_areas = [graph.nodes[n]["area"] for n in lcc
                     if np.isfinite(graph.nodes[n]["area"])]
        if lcc_areas:
            metrics["graph_avg_node_area_lcc"] = float(np.mean(lcc_areas))

    return metrics

def extract_blob_properties(
    df: pd.DataFrame,
    *,
    center_col: str,
    area_col: str,
    radius_col: str,
    bbox_col: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
           Optional[np.ndarray], Tuple[float, float]]:
    """Extracts geometric properties and image size from a blob DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame loaded from a blob data pickle file.
    center_col : str
        Column name for blob centroids (x, y).
    area_col : str
        Column name for blob areas.
    radius_col : str
        Column name for blob radii.
    bbox_col : str
        Column name for bounding boxes (x, y, w, h).

    Returns
    -------
    Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Tuple[float, float]]
        A tuple containing: (centroids, areas, radii, (image_width, image_height)).
        Returns None for arrays if data is missing.
    """
    required_cols = {center_col, area_col, radius_col, bbox_col}
    if not required_cols.issubset(df.columns) or df.empty:
        return None, None, None, (np.nan, np.nan)

    centroids = np.asarray(df[center_col].tolist(), dtype=float)
    areas = df[area_col].to_numpy(dtype=float)
    radii = df[radius_col].to_numpy(dtype=float)

    bbox_array = np.asarray(df[bbox_col].tolist(), dtype=float)
    img_w = float(np.nanmax(bbox_array[:, 2])) if bbox_array.size else np.nan
    img_h = float(np.nanmax(bbox_array[:, 3])) if bbox_array.size else np.nan

    return centroids, areas, radii, (img_w, img_h)

def calculate_all_spatial_metrics(
    df_blobs: pd.DataFrame,
    *,
    graph_method: str | None,
    graph_param: Optional[Union[int, float]] = None,
) -> Dict[str, Any]:
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
    Dict[str, Any]
        A dictionary containing all calculated metrics for the image.
    """
    metrics: Dict[str, Any] = {}
    centroids, areas, radii, (w, h) = extract_blob_properties(
        df_blobs,
        center_col="center",
        area_col="area",
        radius_col="radius",
        bbox_col="bbox",
    )
    img_area = w * h if np.isfinite(w) and np.isfinite(h) else np.nan

    if areas is not None and radii is not None and areas.size > 0:
        metrics.update(
            num_blobs=int(areas.size),
            mean_blob_area=float(areas.mean()),
            median_blob_area=float(np.median(areas)),
            std_blob_area=float(areas.std(ddof=0)),
            total_blob_area=float(areas.sum()),
            mean_blob_radius=float(radii.mean()),
            median_blob_radius=float(np.median(radii)),
        )
    else:
        metrics.update(
            num_blobs=0,
            mean_blob_area=np.nan,
            median_blob_area=np.nan,
            std_blob_area=np.nan,
            total_blob_area=0.0,
            mean_blob_radius=np.nan,
            median_blob_radius=np.nan,
        )

    tba = metrics["total_blob_area"]
    coverage = (100.0 * tba / img_area) if np.isfinite(img_area) and img_area > 0 else np.nan
    metrics["coverage_percentage"] = float(coverage)

    metrics.update(calculate_nnd_stats(centroids))
    metrics.update(calculate_voronoi_stats(centroids))
    metrics.update(
        calculate_graph_metrics(
            centroids, areas, method=graph_method, param=graph_param
        )
    )
    return metrics

def calculate_summary_statistics(
    df: pd.DataFrame,
    group_cols: list[str],
    carry_over_cols: list[str],
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
    group_cols : list[str]
        Columns to group by; must exist in df.
    carry_over_cols : list[str]
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
    valid_groups: list[str] = [c for c in group_cols if c in df.columns]
    if not valid_groups:
        raise ValueError(f"None of the grouping columns {group_cols} exist.")

    carry: list[str] = [c for c in carry_over_cols if c in df.columns]

    num_all: list[str] = list(df.select_dtypes(include="number").columns)
    excl_set: set[str] = set(exclude_numeric_cols or [])
    excl_rx: list[re.Pattern[str]] = [
        re.compile(p) for p in (exclude_numeric_regex or [])
    ]

    def _skip(col: str) -> bool:
        """Return True if col should be excluded from numeric aggregation."""
        if col in valid_groups or col in carry or col in excl_set:
            return True
        return any(r.search(col) for r in excl_rx)

    numeric_cols: list[str] = [c for c in num_all if not _skip(c)]

    if not numeric_cols:
        return df[valid_groups + carry].drop_duplicates().reset_index(drop=True)

    agg_spec: Dict[str, Any] = {c: ["min", "max", "median", "std"] for c in numeric_cols}
    agg_spec.update({c: "first" for c in carry})

    grouped: pd.DataFrame = df.groupby(valid_groups, as_index=False).agg(agg_spec)

    grouped.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0] for col in grouped.columns
    ]
    for c in carry:
        grouped.rename(columns={f"{c}_first": c}, inplace=True)

    return grouped

def process_directory(
    input_dir: Path,
    *,
    mode: str,
    graph_method: str | None,
    graph_param: int | float | None = None,
    time_label: str | None = None,
) -> pd.DataFrame:
    """
    Scans a directory, computes spatial metrics for 
    each file, and returns a DataFrame.

    This function recursively searches for pickle files, 
    parses metadata from their filenames, loads them, 
    and computes a suite of spatial metrics.

    Parameters
    ----------
    input_dir : Path
        The root directory to search for pickle files.
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
    rows: list[Dict[str, Any]] = []

    if mode == "OpenCV":
        glob_pattern = "*_bubble_data.pkl"
        parse_fn = _parse_opencv_filename
        loader_fn = _load_opencv_df
    elif mode == "BubbleSAM":
        glob_pattern = "*_masks_filtered.pkl"
        parse_fn = _parse_bubblesam_filename
        loader_fn = _load_bubblesam_df
    else:
        raise ValueError("Mode must be either 'OpenCV' or 'BubbleSAM'.")

    for pkl_path in input_dir.rglob(glob_pattern):
        metadata = parse_fn(pkl_path.name)
        if not metadata:
            warnings.warn(f"Could not parse metadata from filename: {pkl_path.name}")
            continue
        if time_label:
            metadata["Time"] = time_label
        try:
            df_blobs = loader_fn(pkl_path)
        except (pickle.UnpicklingError, ValueError) as exc:
            warnings.warn(f"Failed to load or parse {pkl_path}: {exc}")
            continue
        metrics = calculate_all_spatial_metrics(
            df_blobs, graph_method=graph_method, graph_param=graph_param
        )
        metrics["image_name"] = pkl_path.with_suffix(".tiff").name
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
    graph_method: str | None,
    graph_param: int | float | None = None,
    composition_csv: Path | None = None,
    cols_to_add: Sequence[str] | None = None,
    group_cols: Sequence[str] | None = None,
    carry_over_cols: Sequence[str] | None = None,
    time_label: str | None = None,
    exclude_numeric_cols: Sequence[str] | None = None,
    exclude_numeric_regex: Sequence[str] | None = None,
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
        The root directory containing the raw per-blob pickle files.
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
    exclude_numeric_cols : Sequence[str] | None
        Exact numeric columns to exclude from aggregation.
    exclude_numeric_regex : Sequence[str] | None
        Regex patterns; matching numeric columns are excluded.

    Returns
    -------
    None
    """
    per_img_df = process_directory(
        input_dir,
        mode=mode,
        graph_method=graph_method,
        graph_param=graph_param,
        time_label=time_label,
    )

    id_cols = ["image_name", "Offset", "Position", "Label", "Class", "Time", "UniqueID"]
    ordered_cols = id_cols + [c for c in per_img_df.columns if c not in id_cols]
    per_img_df = per_img_df[ordered_cols]
    per_image_csv.parent.mkdir(parents=True, exist_ok=True)
    per_img_df.to_csv(per_image_csv, index=False)
    print(f"Per-image metrics saved to: {per_image_csv}")

    if composition_csv and cols_to_add:
        comp_df = pd.read_csv(composition_csv)
        per_img_df = merge_composition_data(
            per_img_df, comp_df, cols_to_add=cols_to_add, merge_key="UniqueID"
        )

    final_group_cols = list(group_cols or ["Group", "Label", "Time", "Class"])
    final_carry_cols = list(carry_over_cols or [])
    agg_df = calculate_summary_statistics(
        per_img_df,
        group_cols=final_group_cols,
        carry_over_cols=final_carry_cols,
        exclude_numeric_cols=exclude_numeric_cols,
        exclude_numeric_regex=exclude_numeric_regex,
    )

    agg_clean_df = _drop_invalid_phase_rows(agg_df, 'Phase_Separation')
    aggregate_csv.parent.mkdir(parents=True, exist_ok=True)
    agg_clean_df.to_csv(aggregate_csv, index=False)
    print(f"Aggregated metrics saved to: {aggregate_csv}")