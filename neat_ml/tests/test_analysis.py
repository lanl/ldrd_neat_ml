from pathlib import Path
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from neat_ml.analysis import data_analysis as da
from pandas.testing import assert_frame_equal
import logging


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
    npt.assert_equal(actual, expected)

@pytest.mark.parametrize("pts, exp, warn_msg",
    [
        # case with sufficient data points to generate
        # voronoi regions
        (
            np.array(
                [[1827.5,  601. ],
                [1117. ,  792. ],
                [1188.5, 1127.5],
                [1189. , 1129. ],
                [1347.5,  174. ],
                [ 179.5, 1591. ],
                [ 804. ,  963.5],
                [1622. , 1698. ]]
            ),
            {
                'mean_voronoi_area': 805822.2719470514,
                'median_voronoi_area': 337884.2409091698,
                'std_voronoi_area': 953547.7664847055,
            },
            None,
        ),
        # case with no finite areas found in voronoi
        # regions
        (
            np.array(
                [[1924.5,  841. ],
                [ 808.5,  485. ],
                [ 306.5,  538. ],
                [2083.5, 1221. ],
                [ 344. ,  633.5],
                [ 724.5, 1047.5]]
            ),
            {},
            "No finite areas found in Voronoi regions",
        ),
    ]
)
def test_calculate_voronoi_stats(caplog, pts, exp, warn_msg):
    """
    test that ``calculate_voronoi_stats`` returns array of
    values from voronoi region calculation OR returns array of nan
    and warns that no finite areas were found in voronoi regions
    """
    caplog.set_level(logging.WARNING)
    actual = da._calculate_voronoi_stats(pts)
    npt.assert_equal(actual, exp)
    if warn_msg is not None:
        assert warn_msg in caplog.text


@pytest.mark.parametrize("method, r_param, k_param, pts, areas, expected",
    [
        (
        # test case where there ARE enough points to perform
        # triangulation
            "delaunay",
            None,
            None,
            None,
            None,
            [10, 21, 4.2, 1.1661903789690602,
            0.5566666666666666, 399.3586811589318, 1, 1.0,510.1],
        ),
        (
        # test case where there ARE NOT enough points to perform
        # triangulation
            "delaunay",
            None,
            None,
            np.array([[1930. , 1947. ],
                     [ 167.5, 1710. ]]),
            np.array([475., 341.]),
            [2, 0, 0.0, 0.0, 2, 0.5, 475.0],
        ),
        # a radius param that only generates a single edge
        (
            "radius",
            100,
            None,
            None,
            None,
            [10, 1, 0.2, 0.4000000000000001,
            0.0, 92.28488500290825, 9, 0.2, 793.0]
        ),
        # a radius param that generates multiple edges
        (
            "radius",
            200,
            None,
            None,
            None,
            [10, 3, 0.6, 0.48989794855663565,
            0.0, 128.97026785384764, 7, 0.2, 156.0]
        ),
        # test case where k=1 nearest neighbors
        (
            "knn", 
            None,
            1,
            None,
            None,
            [10, 6, 1.2, 0.4, 0.0, 217.84017139873353, 4, 0.4, 632.25]
        ),
        # test case where k=3 nearest neighbors (different size output tree)
        (
            "knn", 
            None,
            3,
            None,
            None,
            [10, 20, 4.0, 1.1832159566199232,
            0.6066666666666667, 353.41649782593197, 1, 1.0, 510.1]
        ),
        (
        # test case where the number of input points is less than the
        # provided k value, such that the k value is modified to support
        # the appropriate number of neighbors
            "knn",
            None,
            3,
            np.array([[1930. , 1947. ],
                     [ 167.5, 1710. ]]),
            np.array([475., 341.]),
            [2, 1, 1.0, 0.0, 0.0, 1778.363081600605, 1, 1.0, 408.0],
        ),
    ]
)
def test_calculate_graph_metrics(method, r_param, k_param, pts, areas, expected):
    exp_cols = ['graph_num_nodes', 'graph_num_edges', 'graph_avg_degree',
       'graph_degree_std', 'graph_avg_clustering',
       'graph_avg_neighbor_distance', 'graph_num_components',
       'graph_lcc_node_fraction', 'graph_avg_node_area_lcc']
    if pts is None:
        # generate realistic "points" and "areas" inputs
        rng = np.random.default_rng(0)
        pts = rng.integers(2, 2000, (10,2)) * 0.5
        areas = rng.integers(1, 1000, 10).astype(float)
    elif pts is not None and method == "delaunay":
        # modify the expected columns to remove `graph_avg_clustering` and
        # `graph_avg_neighbor_distance` (not calculated for delaunay method
        # when not enough points for triangulation, stored as NaN by default
        # values in ``calculate_all_spatial_metrics``)
        exp_cols = [col for col in exp_cols 
            if col not in ["graph_avg_clustering", "graph_avg_neighbor_distance"]]
    actual = da._calculate_graph_metrics(pts, areas, method=method, k_param=k_param, r_param=r_param)
    exp_out = dict(zip(exp_cols, expected)) 
    assert actual == exp_out
    

def test_extract_blob_properties(make_dummy_blobs):
    df, expected_center_x, expected_center_y, expected_areas, expected_radii = make_dummy_blobs
    actual_centers, actual_areas, actual_radii, (actual_w, actual_h) = (
        da._extract_blob_properties(
            df,
            center_cols=["center_x", "center_y"],
            area_col="area",
            radius_col="radius",
            bbox_col="bbox",
        )
    )
    expected_centers = np.stack([expected_center_x, expected_center_y], axis=1)
    npt.assert_allclose(actual_centers, expected_centers)
    npt.assert_allclose(actual_areas, expected_areas)
    npt.assert_allclose(actual_radii, expected_radii)
    npt.assert_allclose((actual_w, actual_h), (97.0, 90.0))

def test_calculate_all_spatial_metrics(make_dummy_blobs):
    df, _, _, areas, radii = make_dummy_blobs

    actual = da._calculate_all_spatial_metrics(df, graph_method="delaunay")

    # some sanity checks for outputs
    assert len(actual) == 22
    npt.assert_allclose(actual["mean_blob_area"], areas.mean())
    assert actual["graph_num_nodes"] == 10
    npt.assert_allclose(actual["graph_avg_neighbor_distance"], 29.450361365095052)
    npt.assert_allclose(actual["mean_nnd"], 16.695406977528428)

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
    actual_df = da._load_df(parquet_path, "BubbleSAM")
    expected_center_x = (20.0 + 60.0) / 2
    expected_center_y = (10.0 + 50.0) / 2
    expected_radius = np.sqrt(100.0 / np.pi)

    actual_center_x = actual_df["center_x"].iloc[0]
    actual_center_y = actual_df["center_y"].iloc[0]
    actual_radius = actual_df["radius"].iloc[0]

    npt.assert_allclose(actual_center_x, expected_center_x)
    npt.assert_allclose(actual_center_y, expected_center_y)
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
        r_param=30,
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
    assert df_agg.shape == (1, 94)
    assert df_per.shape == (1, 29)
    
    assert "Phase_Separation" in df_agg.columns
    npt.assert_array_equal(df_agg.columns[:5], group_cols)
    # numerical checks on the per-image df
    npt.assert_allclose(df_per["std_blob_area"], 101.57799958652464)
    npt.assert_allclose(df_per["graph_degree_std"], 0.8717797887081347)
    npt.assert_allclose(df_per["mean_voronoi_area"], 5450.5100956330925)
    npt.assert_allclose(df_per["coverage_percentage"], 38.04123711340206)
    # numerical checks on the aggregated df
    npt.assert_allclose(df_agg["mean_blob_area_max"], 332.1)
    npt.assert_allclose(df_agg["graph_avg_degree_min"], 2.2)
    npt.assert_allclose(df_agg["graph_avg_neighbor_distance_min"], 21.13193381222392)
    npt.assert_allclose(df_agg["mean_nnd_median"], 16.695406977528428) 

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
    test that ``process_parquet_files`` warns on unparsable filenames and
    warns on loader failures (ValueError from ``_load_df``), but still
    returns rows for the good files.
    """
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # a file that contains all the necessary information for
    # returning a dataframe containing a row of calculated metrics
    good = ("offset -1_center_A1_O_Ph_Raw_11111111-"
        f"1111-1111-1111-111111111111_{file_suff}.parquet.gzip")
    pd.DataFrame(
        {
            "area": [10.0],
            "bbox": [(0.0, 0.0, 10.0, 10.0)],
            "center": [(5.0, 5.0)]
        }).to_parquet(
        input_dir / good
    )
    # a parquet file that is not in the correct format and
    # therefore is not readable by the ``_parse_filename`` function
    unparsable = f"weird_{file_suff}.parquet.gzip"
    pd.DataFrame(
        {
            "area": [10.0],
            "bbox": [(0.0, 0.0, 10.0, 10.0)],
            "center": [(5.0, 5.0)],
        }).to_parquet(
        input_dir / unparsable
    )
    # a file that does not contain the appropriate columns
    # for calculating the output metrics with method == bubblesam
    badcontent = ("offset -2_center_A2_O_Ph_Raw_22222222-"
        f"2222-2222-2222-222222222222_{file_suff}.parquet.gzip")
    pd.DataFrame({"wrong_col": [1], "center": [(10.0, 10.0)]}).to_parquet(input_dir / badcontent)
    # a file that does not contain readable parquet data
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
    all_msgs = "\n".join(msgs)
    # check that the appropriate warnings are produced
    # as a result of the various exceptions that can be raised
    assert sum("ArrowInvalid" in m for m in msgs) == 2
    assert sum("ValueError" in m for m in msgs) == n_value_errors
    exp_msgs = [
        "Could not parse metadata from filename",
        "Failed to load or parse",
        "Either the file is corrupted or this is not a parquet file.",
        "Parquet file size is 0 bytes",
    ]
    for exp_msg in exp_msgs:
        assert exp_msg in all_msgs

    assert isinstance(df, pd.DataFrame)
    assert len(df) == len_df


def test_calculate_graph_metrics_bad_method(make_dummy_blobs):
    _, _, pts, areas, _ = make_dummy_blobs
    with pytest.raises(ValueError, match="Invalid input parameters"):
        da._calculate_graph_metrics(pts, areas, method="bad")


@pytest.mark.parametrize("method, k_param, r_param, err_msg",
    [
        ("knn", 1.0, None, "`k_param` must be an integer value"),
        ("knn", 0, None, "`k_param` must be a positive, non-zero integer"),
        ("radius", None, "30", "`r_param` must be either an integer or floating point value"),
        ("radius", None, 0, "`r_param` must be a positive, non-zero value"),
    ]
)
def test_calculate_graph_metrics_bad_params(
    make_dummy_blobs,
    method,
    k_param,
    r_param,
    err_msg,
):
    _, _, pts, areas, _ = make_dummy_blobs
    with pytest.raises(ValueError, match=err_msg):
        da._calculate_graph_metrics(
            pts, areas, method=method, r_param=r_param, k_param=k_param
        )
