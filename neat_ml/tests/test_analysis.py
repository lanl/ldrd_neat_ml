from pathlib import Path
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from neat_ml.analysis import data_analysis as da
from pandas.testing import assert_frame_equal


@pytest.mark.parametrize("pts, expected",
    [
        (
            np.array([[0., 0.], [1., 0.], [2., 0.]]),
            {"mean_nnd": 1.0, "median_nnd": 1.0}
        ),
        (
            np.array([[0., 0.]]),
            {"mean_nnd": np.nan, "median_nnd": np.nan}
        )
    ]
    
)
def test_calculate_nnd_stats(pts, expected):

    actual = da._calculate_nnd_stats(pts)
    actual = pd.DataFrame(actual, index=[0])
    expected = pd.DataFrame(expected, index=[0])
    assert_frame_equal(actual, expected)

@pytest.mark.parametrize("pts, exp",
    [
        # case with sufficient data points to generate
        # voronoi regions
        (
            30,
            [
                0.06852448822053991,
                0.03603173278694283,
                0.07712708084729489,
                1.1255404140935472
            ]
        ),
        # case with insufficient data points to generate
        # voronoi regions
        (
            3,
            [np.nan] * 4,
        ),
    ]
)
def test_calculate_voronoi_stats(pts, exp):
    if isinstance(pts, int):
        rng = np.random.default_rng(seed=0)
        pts = rng.random((pts, 2))
    actual = da._calculate_voronoi_stats(pts)
    exp_out = pd.DataFrame([exp], columns=actual.keys())
    actual_out = pd.DataFrame(actual, index=[0])
    # account for floating point precision errors with absolute tolerance
    assert_frame_equal(actual_out, exp_out, atol=1e-3)


@pytest.mark.parametrize("method, param, expected",
    [
        (
            "delaunay",
            None,
            [10, 21, 4.2, 1.1661903789690602, 1, 1.0,
            0.5566666666666666, 399.3586811589318, 510.1],
        ),
        # a radius param that only generates a single edge
        (
            "radius",
            100,
            [10, 1, 0.2, 0.4000000000000001, 9,
            0.2, 0.0, 92.28488500290825, 793.0]
        ),
        # a radius param that generates multiple edges
        (
            "radius",
            200,
            [10, 3, 0.6, 0.48989794855663565, 7, 0.2,
            0.0, 128.97026785384764, 156.0]
        ),
        # test case where k=1 nearest neighbors
        (
            "knn", 
            1,
            [10, 6, 1.2, 0.4, 4, 0.4, 0.0, 217.84017139873353, 632.25]
        ),
        # test case where k=3 nearest neighbors (different size output tree)
        (
            "knn", 
            3,
            [10, 20, 4.0, 1.1832159566199232, 1, 1.0,
            0.6066666666666667, 353.41649782593197, 510.1]
        ),
    ]
)
def test_calculate_graph_metrics(method, param, expected):
    # generate realistic "points" and "areas" inputs
    rng = np.random.default_rng(0)
    pts = rng.integers(2, 2000, (10,2)) * 0.5
    areas = rng.integers(1, 1000, 10).astype(float)
    actual = da._calculate_graph_metrics(pts, areas, method=method, param=param)
    exp_out = pd.DataFrame([expected], columns=actual.keys())
    actual_out = pd.DataFrame(actual, index=[0])
    assert_frame_equal(actual_out, exp_out)
    

def test_extract_blob_properties(make_dummy_blobs):
    df, expected_centres, expected_areas, expected_radii = make_dummy_blobs
    actual_centres, actual_areas, actual_radii, (actual_w, actual_h) = (
        da._extract_blob_properties(
            df,
            center_col="center",
            area_col="area",
            radius_col="radius",
            bbox_col="bbox",
        )
    )

    npt.assert_array_equal(actual_centres, expected_centres)
    npt.assert_array_equal(actual_areas, expected_areas)
    npt.assert_array_equal(actual_radii, expected_radii)
    npt.assert_allclose((actual_w, actual_h), (100.0, 100.0))

def test_calculate_all_spatial_metrics(make_dummy_blobs):
    df, _, areas, radii = make_dummy_blobs

    actual = da._calculate_all_spatial_metrics(df,graph_method="delaunay")

    expected = {
        "num_blobs": 4,
        "mean_blob_area": areas.mean(),
        "median_blob_area": np.median(areas),
        "std_blob_area": areas.std(ddof=0),
        "total_blob_area": areas.sum(),
        "mean_blob_radius": radii.mean(),
        "median_blob_radius": np.median(radii),
        "coverage_percentage": 4.2,
        "mean_nnd": 80.0,
        "median_nnd": 80.0,
        "graph_num_nodes": 4,
        "graph_num_edges": 5,
    }
    actual_df = pd.DataFrame(actual, index=[0])
    actual_df = actual_df[expected.keys()]
    expected_df = pd.DataFrame(expected, index=[0])
    assert_frame_equal(actual_df, expected_df)

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
    out = da._calculate_all_spatial_metrics(df_empty, graph_method="delaunay")

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
    parquet_path = tmp_path / "mock_masks_filtered.parquet.gzip"
    df_original.to_parquet(parquet_path)
    actual_df = da._load_df(parquet_path, "bubblesam")
    expected_center = ((20.0 + 60.0) / 2, (10.0 + 50.0) / 2)
    expected_radius = np.sqrt(100.0 / np.pi)

    actual_center = actual_df["center"].iloc[0]
    actual_radius = actual_df["radius"].iloc[0]

    npt.assert_allclose(actual_center, expected_center)
    npt.assert_allclose(actual_radius, expected_radius)

@pytest.mark.parametrize("row, exp",
    [
        ("Phase_Separation", 2),
        ("NonExistentCol", 4),
    ]
)
def test_drop_invalid_phase_rows(row, exp):
    df = pd.DataFrame({
        "Phase_Separation": [True, np.nan, False, ""],
        "other": [1, 2, 3, 4],
    })
    actual = da._drop_invalid_phase_rows(df, row)
    assert actual.shape == (exp, 2)

def test_voronoi_qhull_error():
    """
    Test that a warning is issued when Voronoi calculation fails.
    """
    # four colinear points
    dummy_points = np.array([1, 2, 3, 4])
    with pytest.warns(UserWarning, match="Voronoi calculation failed"):
        actual = da._calculate_voronoi_stats(dummy_points)
        npt.assert_array_equal(list(actual.values()), [np.nan] * 4)

@pytest.mark.parametrize("input_data, expected",
    [
        (None, 0),
        ("points", 4)
    ]
)
def test_calculate_graph_metrics_failures(square_points, input_data, expected):
    if input_data is not None:
        input_1, input_2 = square_points
        input_2 = input_2[:2]
    else:
        input_1 = input_2 = input_data
    actual_none = da._calculate_graph_metrics(input_1, input_2, method="Delaunay")
    assert actual_none["graph_num_nodes"] == expected
    assert np.isnan(actual_none["graph_avg_degree"])

@pytest.mark.parametrize("input_data",
    ["empty_df", "center"]
)
def test_extract_blob_properties_failures(make_dummy_blobs, input_data):
    if input_data == "center":
        df, _, _, _ = make_dummy_blobs
        input_df = df.drop(columns=['center'])
    else:
        input_df = pd.DataFrame()
    c, a, r, (w, h) = da._extract_blob_properties(
        input_df,
        center_col="center",
        area_col="area",
        radius_col="radius",
        bbox_col="bbox",
    )
    assert all(x.size == 0 for x in (a, r, c))
    assert np.isnan(w) and np.isnan(h)

@pytest.mark.parametrize(
    "fname, expected, method",
    [
        (
            ("offset -5_bottom_A2_O_Ph_Raw_163c48ec-5ec9"
             "-4b1c-b304-ea40e77f0780_bubble_data.parquet.gzip"),
            {
                "UniqueID": "163c48ec-5ec9-4b1c-b304-ea40e77f0780",
                "Class": "Ph",
                "Offset": -5,
                "Position": "bottom",
                "Label": "A2"
            },
            "opencv",
        ),
        (
            ("offset -5_bottom_A1_O_Ph_Raw_b96c0d64-03fd"
             "-4285-824d-e82eafedce90_masks_filtered.parquet.gzip"),
            {
                "UniqueID": "b96c0d64-03fd-4285-824d-e82eafedce90",
                "Class": "Ph",
                "Offset": -5,
                "Position": "bottom",
                "Label": "A1"
            },
            "bubblesam",
        ),
        ("invalid_filename.parquet.gzip", {}, "bubblesam"),
    ],
)
def test_filename_parsers(fname, expected, method):
    actual = da._parse_filename(fname, method)
    assert actual == expected

def test_calculate_summary_statistics():
    df = pd.DataFrame({
        "Group": ["A", "A", "B"],
        "Time": [0, 0, 1],
        "metric": [10.0, 20.0, 30.0],
        "other_metric": [5.0, 10.0, 15.0],
        "exclude_metric": [1.0, 2.0, 3.0],
        "info": ["x", "x", "y"],
    })
    actual = da._calculate_summary_statistics(
        df,
        group_cols=["Group"],
        carry_over_cols=["info"],
        exclude_numeric_cols=["other_metric"],
        exclude_numeric_regex=["exclude"],
    )
    assert len(actual) == 2
    assert "metric_median" in actual.columns
    assert "info" in actual.columns
    with pytest.raises(ValueError, match="None of the grouping columns"):
        da._calculate_summary_statistics(df, ["BadCol"], [])


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

    out = da._calculate_summary_statistics(
        df,
        group_cols=["Group", "Label"],
        carry_over_cols=["Other"],
        exclude_numeric_cols=None,
        exclude_numeric_regex=[r"^graph_.*"],
    )

    expected = df[["Group", "Label", "Other"]].drop_duplicates().reset_index(drop=True)
    assert_frame_equal(out, expected)

def test_merge_composition_data():
    summary_df = pd.DataFrame({"UniqueID": ["id1"], "metric": [10]})
    comp_df = pd.DataFrame({"UniqueID": ["id1"], "Phase": [True]})
    actual = da._merge_composition_data(
        summary_df, 
        comp_df, 
        cols_to_add=["Phase"],
        merge_key="UniqueID"
    )
    assert "Phase" in actual.columns
    with pytest.raises(ValueError, match="not found in summary_df"):
        da._merge_composition_data(
            summary_df.drop(columns=["UniqueID"]),
            comp_df,
            cols_to_add=["Phase"],
            merge_key="UniqueID"
        )

@pytest.mark.parametrize("mode, err, err_msg",
    [
        ("BadMode", ValueError, "Mode must be either"),
        ("OpenCV", FileNotFoundError, "No valid files were processed"),
    ]
)
def test_process_parquet_files_errors(
    tmp_path: Path,
    mode,
    err,
    err_msg,
):
    with pytest.raises(err, match=err_msg):
        da._process_parquet_files(tmp_path, mode=mode, graph_method="delaunay")

@pytest.mark.parametrize("mode, tiff_name", 
    [
        (
            "OpenCV",
            (
                'offset -5_bottom_A2_O_Ph_Raw_163c48ec-5ec9'
                '-4b1c-b304-ea40e77f0780_bubble_data.tiff'
            ),
        ),
        (
            "BubbleSAM",
            (
                'offset -5_bottom_A1_O_Ph_Raw_b96c0d64-03fd'
                '-4285-824d-e82eafedce90_masks_filtered.tiff'
            ),
        )

    ]
)
def test_full_analysis_pipeline(
    mock_dir,
    mode,
    tiff_name,
):
    input_dir, output_dir, comp_csv = mock_dir
    per_image_csv = output_dir / f"per_image_{mode}.csv"
    aggregate_csv = output_dir / f"aggregate_{mode}.csv"
    group_cols = ["Group", "Label", "Time", "Class", "Offset"]

    da.full_analysis(
        input_dir=input_dir,
        per_image_csv=per_image_csv,
        aggregate_csv=aggregate_csv,
        mode=mode,
        graph_method="radius",
        composition_csv=comp_csv,
        cols_to_add=["Phase_Separation", "Group"],
        time_label="1st",
        group_cols=group_cols,
        carry_over_cols=["Phase_Separation"],
    )

    assert per_image_csv.exists()
    assert aggregate_csv.exists()
    df_agg = pd.read_csv(aggregate_csv)
    df_per = pd.read_csv(per_image_csv)
    assert df_per["image_name"].item() == tiff_name
    assert df_agg.shape == (1, 98)
    assert df_per.shape == (1, 30)
    
    assert "Phase_Separation" in df_agg.columns
    assert "Offset" in df_agg.columns
    npt.assert_array_equal(df_agg.columns[:5], group_cols)

def test_merge_composition_data_missing_cols_message():
    """Ensure clear error text when requested cols are missing in composition_df."""
    summary_df = pd.DataFrame({"UniqueID": ["id1"], "metric": [42]})
    comp_df = pd.DataFrame({"UniqueID": ["id1"]})  # 'Phase' is missing

    with pytest.raises(
        ValueError, match=r"Columns \['Phase'\] not found in composition_df\."
    ):
        da._merge_composition_data(
            summary_df,
            comp_df,
            cols_to_add=["Phase"],
            merge_key="UniqueID",
        )


@pytest.mark.parametrize("file_suff, method, n_value_errors, len_df",
    [
        ("masks_filtered", "BubbleSAM", 1, 1),
        # should only throw a value error when
        # bubblesam is missing area/bbox columns
        ("bubble_data", "OpenCV", 0, 2),
    ]
)
def test_process_parquet_files_warns_and_continues_bubblesam(
    tmp_path: Path,
    file_suff,
    method,
    n_value_errors,
    len_df,
):
    """
    test that ``process_parquet_files`` warns on unparseable filenames and
    warns on loader failures (ValueError from ``_load_df``), but still
    returns rows for the good files.
    """
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # a file that contains all the necessary information for
    # returning a dataframe containing a row of calculated metrics
    good = ("offset -1_center_A1_O_Ph_Raw_11111111-"
        f"1111-1111-1111-111111111111_{file_suff}.parquet.gzip")
    pd.DataFrame({"area": [10.0], "bbox": [(0.0, 0.0, 10.0, 10.0)]}).to_parquet(
        input_dir / good
    )
    # a parquet file that is not in the correct format and
    # therefore is not readable by the ``_parqe_filename`` function
    unparseable = f"weird_{file_suff}.parquet.gzip"
    pd.DataFrame({"area": [10.0], "bbox": [(0.0, 0.0, 10.0, 10.0)]}).to_parquet(
        input_dir / unparseable
    )
    # a file that does not contain the appropriate columns
    # for calculating the output metrics with method == bubblesam
    badcontent = ("offset -2_center_A2_O_Ph_Raw_22222222-"
        f"2222-2222-2222-222222222222_{file_suff}.parquet.gzip")
    pd.DataFrame({"wrong_col": [1]}).to_parquet(input_dir / badcontent)
    # a file that does not readable parquet data
    arrow_invalid = ("offset -1_center_A3_O_Ph_Raw_33333333-"
        f"3333-3333-3333-333333333333_{file_suff}.parquet.gzip")
    with open(input_dir / arrow_invalid, "w") as f:
        f.write("ArrowInvalid text file")
    # a file that is completely empty
    arrow_io = ("offset -1_center_A3_O_Ph_Raw_44444444-"
        f"4444-4444-4444-444444444444_{file_suff}.parquet.gzip")
    with open(input_dir / arrow_io, "w") as f:
        pass

    with pytest.warns(UserWarning) as record:
        df = da._process_parquet_files(input_dir, mode=method, graph_method="delaunay")

    msgs = [str(w.message) for w in record.list]
    # check that the appropriate warnings are produced
    # as a result of the various exceptions that can be raised
    assert sum("ArrowInvalid" in m for m in msgs) == 2
    assert sum("ValueError" in m for m in msgs) == n_value_errors
    assert any("Could not parse metadata from filename" in m for m in msgs)
    assert any("Failed to load or parse" in m for m in msgs)
    assert any("Either the file is corrupted or this is not a parquet file." in m for m in msgs)
    assert any("Parquet file size is 0 bytes" in m for m in msgs)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == len_df

def test_calculate_nnd_stats_warns_on_no_finite_distances():
    """
    Force the 'No finite neighbor distances found.' by providing
    extremely large value to ``calculate_nnd_stats`` and check that
    return df contains nan values.
    """
    large_val = 1e308

    # These points are finite, but the distance between them
    # (2e308) cannot be represented and becomes 'inf'.
    pts = np.array([
        [large_val, large_val],
        [-large_val, -large_val]
    ])

    with pytest.warns(
        UserWarning, match="NND calculation failed: No finite neighbor distances found"
    ):
        res = da._calculate_nnd_stats(pts)

    assert np.isnan(res["mean_nnd"]) and np.isnan(res["median_nnd"])


def test_calculate_graph_metrics_warns_on_exception():
    """
    Cause a graph-construction failure and ensure it warns but still returns
    sensible metrics (nodes present, no edges, multiple components).
    """
    pts = np.array([
        [np.nan, np.nan],
        [1, 1],
        [2, 2],
        [3, 3]
    ])
    areas = np.ones(4)

    with pytest.warns(UserWarning, match="Graph construction"):
        res = da._calculate_graph_metrics(pts, areas, method="delaunay")

    assert res["graph_num_nodes"] == 4
    assert res["graph_num_edges"] == 0
    assert res["graph_num_components"] == 4
    assert res["graph_avg_degree"] == 0.0


def test_calculate_graph_metrics_bad_method(make_dummy_blobs):
    _, pts, areas, _ = make_dummy_blobs
    with pytest.raises(ValueError, match="Invalid input parameters"):
        da._calculate_graph_metrics(pts, areas, method="bad")
