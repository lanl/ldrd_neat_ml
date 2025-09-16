from pathlib import Path
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from neat_ml.analysis import data_analysis as da

from scipy.spatial import QhullError

def _make_dummy_blobs():
    """A tiny 4-blob square that is useful across multiple tests."""
    centres = [(10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0)]
    areas   = [100.0, 120.0, 110.0,  90.0]
    radii   = [5.0] * 4
    bboxes  = [(0.0, 0.0, 100.0, 100.0)] * 4

    df = pd.DataFrame(
        {"center": centres, "area": areas, "radius": radii, "bbox": bboxes}
    )
    return df, np.asarray(centres, float), np.asarray(areas, float), np.asarray(radii, float)

@pytest.fixture()
def square_points():
    """Return four points forming a unit square plus helper arrays."""
    pts = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    areas = np.full(4, 1.0)
    return pts, areas

@pytest.fixture
def mock_dir(tmp_path: Path):
    """Creates a mock directory structure for end-to-end pipeline testing."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    ocv_fname = "offset -5_bottom_A2_O_Ph_Raw_163c48ec-5ec9-4b1c-b304-ea40e77f0780_bubble_data.pkl"
    df_ocv, _, _, _ = _make_dummy_blobs()
    df_ocv.to_pickle(input_dir / ocv_fname)

    bsam_fname = "offset -5_bottom_A1_O_Ph_Raw_b96c0d64-03fd-4285-824d-e82eafedce90_masks_filtered.pkl"
    df_bsam = pd.DataFrame({"area": [10.0], "bbox": [(0, 0, 10, 10)]})
    df_bsam.to_pickle(input_dir / bsam_fname)

    comp_df = pd.DataFrame({
        "UniqueID": ["163c48ec-5ec9-4b1c-b304-ea40e77f0780", "b96c0d64-03fd-4285-824d-e82eafedce90"],
        "Phase_Separation": [True, False],
        "Group": ["G1", "G2"],
    })
    comp_csv = tmp_path / "composition.csv"
    comp_df.to_csv(comp_csv, index=False)

    return input_dir, output_dir, comp_csv

def test_calculate_nnd_stats():
    pts = np.array([[0., 0.], [1., 0.], [2., 0.]])

    actual = da.calculate_nnd_stats(pts)
    expected = {"mean_nnd": 1.0, "median_nnd": 1.0}

    npt.assert_allclose(
        [actual["mean_nnd"], actual["median_nnd"]],
        [expected["mean_nnd"], expected["median_nnd"]],
    )

    actual_nan = da.calculate_nnd_stats(np.array([[0., 0.]]))
    expected_nan = {"mean_nnd": np.nan, "median_nnd": np.nan}

    npt.assert_array_equal(
        [actual_nan["mean_nnd"], actual_nan["median_nnd"]],
        [expected_nan["mean_nnd"], expected_nan["median_nnd"]],
    )

def test_calculate_voronoi_stats():
    rng = np.random.default_rng(seed=0)
    pts = rng.random((30, 2))
    actual = da.calculate_voronoi_stats(pts)

    for key in ("mean_voronoi_area", "median_voronoi_area",
                "std_voronoi_area",  "cv_voronoi_area"):
        expected_positive = 0.0
        npt.assert_array_less(expected_positive, actual[key])
        npt.assert_equal(np.isfinite(actual[key]), True)

    expected_cv = actual["std_voronoi_area"] / actual["mean_voronoi_area"]
    npt.assert_allclose(actual["cv_voronoi_area"], expected_cv, rtol=1e-7)
    actual_tiny = da.calculate_voronoi_stats(pts[:3])
    expected_tiny = [np.nan] * 4
    npt.assert_array_equal(list(actual_tiny.values()), expected_tiny)


def test_calculate_graph_metrics_delaunay():
    _, pts, areas, _ = _make_dummy_blobs()
    actual = da.calculate_graph_metrics(pts, areas, method="delaunay")

    expected = {
        "graph_num_nodes":           4,
        "graph_num_edges":           5,
        "graph_avg_degree":          2.5,
        "graph_degree_std":          0.5,
        "graph_num_components":      1,
        "graph_lcc_node_fraction":   1.0,
        "graph_avg_node_area_lcc":   areas.mean(),
        "graph_avg_neighbor_distance": 86.6274,
        "graph_avg_clustering":      0.8333333,
    }

    npt.assert_equal(actual["graph_num_nodes"], expected["graph_num_nodes"])
    npt.assert_equal(actual["graph_num_edges"], expected["graph_num_edges"])
    npt.assert_allclose(actual["graph_avg_degree"], expected["graph_avg_degree"])
    npt.assert_allclose(actual["graph_degree_std"], expected["graph_degree_std"])
    npt.assert_equal(actual["graph_num_components"], expected["graph_num_components"])
    npt.assert_allclose(actual["graph_lcc_node_fraction"],
                        expected["graph_lcc_node_fraction"])
    npt.assert_allclose(actual["graph_avg_node_area_lcc"],
                        expected["graph_avg_node_area_lcc"])
    npt.assert_allclose(actual["graph_avg_neighbor_distance"],
                        expected["graph_avg_neighbor_distance"], rtol=1e-4)
    npt.assert_allclose(actual["graph_avg_clustering"],
                        expected["graph_avg_clustering"], rtol=1e-6)
    
def test_calculate_graph_metrics_radius(square_points):
    pts, areas = square_points
    r = 1.1
    actual_metrics = da.calculate_graph_metrics(pts, areas, method="radius", param=r)
    expected_edges = 4
    expected_avg_degree = 2.0
    expected_degree_std = 0.0
    expected_avg_nbr_dist = 1.0
    expected_components = 1

    npt.assert_equal(actual_metrics["graph_num_edges"], expected_edges)
    npt.assert_allclose(actual_metrics["graph_avg_degree"], expected_avg_degree)
    npt.assert_allclose(actual_metrics["graph_degree_std"], expected_degree_std)
    npt.assert_allclose(actual_metrics["graph_avg_neighbor_distance"],
                        expected_avg_nbr_dist)
    npt.assert_equal(actual_metrics["graph_num_components"], expected_components)

def test_calculate_graph_metrics_knn(square_points):
    pts, areas = square_points
    actual_metrics = da.calculate_graph_metrics(pts, areas, method="knn", param=1)
    expected_nodes = 4
    expected_edges = 3
    expected_avg_degree = 1.5
    expected_degree_std = 0.5
    npt.assert_equal(actual_metrics["graph_num_nodes"], expected_nodes)
    npt.assert_equal(actual_metrics["graph_num_edges"], expected_edges)
    npt.assert_allclose(actual_metrics["graph_avg_degree"], expected_avg_degree)
    npt.assert_allclose(actual_metrics["graph_degree_std"], expected_degree_std)

def test_extract_blob_properties():
    df, expected_centres, expected_areas, expected_radii = _make_dummy_blobs()
    actual_centres, actual_areas, actual_radii, (actual_w, actual_h) = (
        da.extract_blob_properties(
            df,
            center_col="center",
            area_col="area",
            radius_col="radius",
            bbox_col="bbox",
        )
    )

    npt.assert_array_equal(actual_centres, expected_centres)
    npt.assert_array_equal(actual_areas,   expected_areas)
    npt.assert_array_equal(actual_radii,   expected_radii)
    npt.assert_allclose((actual_w, actual_h), (100.0, 100.0))

def test_calculate_all_spatial_metrics():
    df, _, areas, radii = _make_dummy_blobs()

    actual = da.calculate_all_spatial_metrics(df,graph_method="delaunay")

    expected = {
        "num_blobs":          4,
        "mean_blob_area":     areas.mean(),
        "median_blob_area":   np.median(areas),
        "std_blob_area":      areas.std(ddof=0),
        "total_blob_area":    areas.sum(),
        "mean_blob_radius":   radii.mean(),
        "median_blob_radius": np.median(radii),
        "coverage_percentage": 4.2,
        "mean_nnd":            80.0,
        "median_nnd":          80.0,
        "graph_num_nodes":     4,
        "graph_num_edges":     5,
    }

    npt.assert_equal(actual["num_blobs"], expected["num_blobs"])
    npt.assert_equal(actual["graph_num_nodes"], expected["graph_num_nodes"])
    npt.assert_equal(actual["graph_num_edges"], expected["graph_num_edges"])

    for key in ("mean_blob_area", "median_blob_area", "std_blob_area",
                "total_blob_area", "mean_blob_radius", "median_blob_radius",
                "coverage_percentage", "mean_nnd", "median_nnd"):
        npt.assert_allclose(actual[key], expected[key])

@pytest.mark.parametrize(
    "df_empty",
    [
        pd.DataFrame(columns=["center", "area", "radius", "bbox"]),
        pd.DataFrame(
            {
                "center": [(10.0, 20.0), (30.0, 40.0)],
                "bbox": [(0, 0, 100, 100), (5, 5, 200, 150)],
                # "area" and "radius" intentionally omitted
            }
        ),
    ],
    ids=["empty_df_with_required_cols", "missing_required_cols"],
)
def test_calculate_all_spatial_metrics_else_branch_defaults(df_empty):
    out = da.calculate_all_spatial_metrics(df_empty, graph_method="delaunay")

    assert out["num_blobs"] == 0
    assert out["total_blob_area"] == 0.0

    for key in (
        "mean_blob_area",
        "median_blob_area",
        "std_blob_area",
        "mean_blob_radius",
        "median_blob_radius",
        "coverage_percentage",
        "mean_nnd",
        "median_nnd",
        "graph_avg_degree",
        "graph_degree_std",
        "graph_num_components",
        "graph_lcc_node_fraction",
        "graph_avg_clustering",
        "graph_avg_neighbor_distance",
        "graph_avg_node_area_lcc",
    ):
        npt.assert_allclose(out[key], np.nan, equal_nan=True)

    npt.assert_array_equal(
        np.array([out["graph_num_nodes"], out["graph_num_edges"]]),
        np.array([0, 0]),
    )


def test_load_bubblesam_df(tmp_path: Path):
    df_original = pd.DataFrame(
        {
            "area": [100.0],
            "bbox": [(10.0, 20.0, 50.0, 60.0)],
        }
    )
    pkl_path = tmp_path / "mock_masks_filtered.pkl"
    df_original.to_pickle(pkl_path)
    actual_df = da._load_bubblesam_df(pkl_path)
    expected_center = ((20.0 + 60.0) / 2, (10.0 + 50.0) / 2)
    expected_radius = np.sqrt(100.0 / np.pi)

    actual_center = actual_df["center"].iloc[0]
    actual_radius = actual_df["radius"].iloc[0]

    npt.assert_allclose(actual_center, expected_center)
    npt.assert_allclose(actual_radius, expected_radius)

def test_drop_invalid_phase_rows():
    df = pd.DataFrame({
        "Phase_Separation": [True, np.nan, False, ""],
        "other": [1, 2, 3, 4],
    })
    actual = da._drop_invalid_phase_rows(df, "Phase_Separation")
    npt.assert_equal(len(actual), 2)
    actual_no_col = da._drop_invalid_phase_rows(df, "NonExistentCol")
    npt.assert_equal(len(actual_no_col), 4)

def test_voronoi_qhull_error(mocker):
    """
    Test that a warning is issued when Voronoi calculation fails.
    """
    mocker.patch(
        "neat_ml.analysis.data_analysis.Voronoi",
        side_effect=QhullError("mocked qhull error")
    )
    dummy_points = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    
    with pytest.warns(UserWarning, match="Voronoi calculation failed: mocked qhull error"):
        actual = da.calculate_voronoi_stats(dummy_points)
        npt.assert_array_equal(list(actual.values()), [np.nan] * 4)

def test_calculate_graph_metrics_failures(square_points):
    pts, areas = square_points
    actual_none = da.calculate_graph_metrics(None, None, method="Delaunay")
    npt.assert_equal(actual_none["graph_num_nodes"], 0)
    assert np.isnan(actual_none["graph_avg_degree"])
    actual_mismatch = da.calculate_graph_metrics(pts, areas[:2], method="Delaunay")
    npt.assert_equal(actual_mismatch["graph_num_nodes"], 4)
    assert np.isnan(actual_mismatch["graph_avg_degree"])

def test_extract_blob_properties_failures():
    df, _, _, _ = _make_dummy_blobs()
    c, a, r, (w, h) = da.extract_blob_properties(
        pd.DataFrame(),
        center_col="center",
        area_col="area",
        radius_col="radius",
        bbox_col="bbox",
    )
    assert c is None and a is None and r is None
    assert np.isnan(w) and np.isnan(h)
    c, a, r, (w, h) = da.extract_blob_properties(
        df.drop(columns=["center"]),
        center_col="center",
        area_col="area",
        radius_col="radius",
        bbox_col="bbox",
    )
    assert c is None and a is None and r is None

@pytest.mark.parametrize(
    "fname,expected",
    [
        (
            "offset -5_bottom_A2_O_Ph_Raw_163c48ec-5ec9-4b1c-b304-ea40e77f0780_bubble_data.pkl",
            {"UniqueID": "163c48ec-5ec9-4b1c-b304-ea40e77f0780", "Class": "Ph", "Offset": -5, "Position": "bottom", "Label": "A2"},
        ),
        (
            "offset -5_bottom_A1_O_Ph_Raw_b96c0d64-03fd-4285-824d-e82eafedce90_masks_filtered.pkl",
            {"UniqueID": "b96c0d64-03fd-4285-824d-e82eafedce90", "Class": "Ph", "Offset": -5, "Position": "bottom", "Label": "A1"},
        ),
        ("invalid_filename.pkl", {}),
    ],
)
def test_filename_parsers(fname, expected):
    if "bubble_data" in fname:
        actual = da._parse_opencv_filename(fname)
    else:
        actual = da._parse_bubblesam_filename(fname)
    npt.assert_equal(actual, expected)

def test_load_bubblesam_df_error(tmp_path: Path):
    df_bad = pd.DataFrame({"wrong_col": [1]})
    pkl_path = tmp_path / "bad.pkl"
    df_bad.to_pickle(pkl_path)
    with pytest.raises(ValueError, match="missing 'area' or 'bbox'"):
        da._load_bubblesam_df(pkl_path)

def test_calculate_summary_statistics():
    df = pd.DataFrame({
        "Group": ["A", "A", "B"],
        "Time": [0, 0, 1],
        "metric": [10.0, 20.0, 30.0],
        "info": ["x", "x", "y"],
    })
    actual = da.calculate_summary_statistics(
        df, group_cols=["Group"], carry_over_cols=["info"]
    )
    npt.assert_equal(len(actual), 2)
    assert "metric_median" in actual.columns
    assert "info" in actual.columns
    with pytest.raises(ValueError, match="None of the grouping columns"):
        da.calculate_summary_statistics(df, ["BadCol"], [])


def test_returns_groups_and_carry_when_all_numeric_excluded_by_regex():
    """
    numeric_cols exists in df but is emptied by exclude_numeric_regex.
    The function should return the deduplicated group+carry view.
    """
    df = pd.DataFrame(
        {
            "Group": ["G1", "G1", "G2"],
            "Label": ["A", "A", "B"],
            "Other": ["x", "x", "y"],
            "graph_num_nodes": [10, 10, 20],
            "graph_num_edges": [15, 15, 30],
        }
    )

    out = da.calculate_summary_statistics(
        df,
        group_cols=["Group", "Label"],
        carry_over_cols=["Other"],
        exclude_numeric_cols=None,
        exclude_numeric_regex=[r"^graph_.*"],
    )

    expected = df[["Group", "Label", "Other"]].drop_duplicates().reset_index(drop=True)

    assert list(out.columns) == ["Group", "Label", "Other"]
    npt.assert_array_equal(out.to_numpy(), expected.to_numpy())

def test_merge_composition_data():
    summary_df = pd.DataFrame({"UniqueID": ["id1"], "metric": [10]})
    comp_df = pd.DataFrame({"UniqueID": ["id1"], "Phase": [True]})
    actual = da.merge_composition_data(
        summary_df, 
        comp_df, 
        cols_to_add=["Phase"],
        merge_key="UniqueID"
    )
    assert "Phase" in actual.columns
    with pytest.raises(ValueError, match="not found in summary_df"):
        da.merge_composition_data(
            summary_df.drop(columns=["UniqueID"]),
            comp_df,
            cols_to_add=["Phase"],
            merge_key="UniqueID"
        )

def test_process_directory_errors(tmp_path: Path):
    with pytest.raises(ValueError, match="Mode must be either"):
        da.process_directory(tmp_path, mode="BadMode", graph_method="delaunay")
    with pytest.raises(FileNotFoundError, match="No valid files were processed"):
        da.process_directory(tmp_path, mode="OpenCV", graph_method="delaunay")

@pytest.mark.parametrize("mode", ["OpenCV", "BubbleSAM"])
def test_full_analysis_pipeline(mock_dir, mode):
    input_dir, output_dir, comp_csv = mock_dir
    per_image_csv = output_dir / f"per_image_{mode}.csv"
    aggregate_csv = output_dir / f"aggregate_{mode}.csv"

    da.full_analysis(
        input_dir=input_dir,
        per_image_csv=per_image_csv,
        aggregate_csv=aggregate_csv,
        mode=mode,
        graph_method="radius",
        composition_csv=comp_csv,
        cols_to_add=["Phase_Separation", "Group"],
        time_label="1st",
        group_cols=["Group", "Label", "Time", "Class", "Offset"],
        carry_over_cols=["Phase_Separation"],
    )

    assert per_image_csv.exists()
    assert aggregate_csv.exists()
    df_agg = pd.read_csv(aggregate_csv)
    
    assert "Phase_Separation" in df_agg.columns
    assert "Offset" in df_agg.columns

def test_merge_composition_data_missing_cols_message():
    """1) Ensure clear error text when requested cols are missing in composition_df."""
    summary_df = pd.DataFrame({"UniqueID": ["id1"], "metric": [42]})
    comp_df = pd.DataFrame({"UniqueID": ["id1"]})  # 'Phase' is missing

    with pytest.raises(
        ValueError, match=r"Columns \['Phase'\] not found in composition_df\."
    ):
        da.merge_composition_data(
            summary_df,
            comp_df,
            cols_to_add=["Phase"],
            merge_key="UniqueID",
        )


def test_process_directory_warns_and_continues_bubblesam(tmp_path: Path):
    """
    2) process_directory:
       - warns on unparseable filenames, AND
       - warns on loader failures (ValueError from _load_bubblesam_df),
       - but still returns rows for the good files.
    """
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    good = "offset -1_center_A1_O_Ph_Raw_11111111-1111-1111-1111-111111111111_masks_filtered.pkl"
    pd.DataFrame({"area": [10.0], "bbox": [(0.0, 0.0, 10.0, 10.0)]}).to_pickle(
        input_dir / good
    )
    unparseable = "weird_masks_filtered.pkl"
    pd.DataFrame({"area": [10.0], "bbox": [(0.0, 0.0, 10.0, 10.0)]}).to_pickle(
        input_dir / unparseable
    )
    badcontent = "offset -2_center_A2_O_Ph_Raw_22222222-2222-2222-2222-222222222222_masks_filtered.pkl"
    pd.DataFrame({"wrong_col": [1]}).to_pickle(input_dir / badcontent)

    with pytest.warns(UserWarning) as record:
        df = da.process_directory(input_dir, mode="BubbleSAM",graph_method="delaunay")

    msgs = [str(w.message) for w in record.list]
    assert any("Could not parse metadata from filename" in m for m in msgs)
    assert any("Failed to load or parse" in m for m in msgs)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1


def test_calculate_nnd_stats_warns_on_no_finite_distances(mocker):
    """
    3) Force the 'No finite neighbor distances found.' path:
       Patch KDTree.query to return all-NaN distances so the function
       raises internally then catches and warns, returning NaNs.
    """
    pts = np.array([[0.0, 0.0], [1.0, 0.0]])
    distances = np.full((2, 2), np.nan)
    indices = np.zeros((2, 2), dtype=int)

    mocker.patch.object(da.KDTree, "query", return_value=(distances, indices))

    with pytest.warns(
        UserWarning, match="NND calculation failed: No finite neighbor distances found."
    ):
        res = da.calculate_nnd_stats(pts)

    assert np.isnan(res["mean_nnd"]) and np.isnan(res["median_nnd"])


def test_calculate_graph_metrics_warns_on_exception(mocker):
    """
    4) Cause a graph-construction failure and ensure it warns but still returns
       sensible metrics (nodes present, no edges, multiple components).
    """
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    areas = np.ones(4)

    mocker.patch(
        "neat_ml.analysis.data_analysis.Delaunay",
        side_effect=RuntimeError("boom"),
    )
    with pytest.warns(UserWarning, match=r"Graph construction \(delaunay\) failed: boom"):
        res = da.calculate_graph_metrics(pts, areas, method="delaunay")

    assert res["graph_num_nodes"] == 4
    assert res["graph_num_edges"] == 0
    assert res["graph_num_components"] == 4
    assert res["graph_avg_degree"] == 0.0
