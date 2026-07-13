from pathlib import Path
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_array_equal
import pandas as pd
import pytest
from neat_ml.analysis import data_analysis as da
from pandas.testing import assert_frame_equal, assert_series_equal
import logging
from typing import Literal


@pytest.mark.parametrize("pts, expected, img_hyp",
    [
        (
            np.array([[0., 0.], [1., 0.], [2., 0.]]),
            {"mean_nnd": 1.0, "median_nnd": 1.0},
            3,
        ),
    ]
    
)
def test_calculate_nnd_stats(pts, expected, img_hyp):
    actual = da._calculate_nnd_stats(pts, img_hyp)
    assert_equal(actual, expected)

@pytest.mark.parametrize("pts, exp, warn_msg",
    [
        # case with sufficient data points to generate
        # voronoi regions
        (
            np.array(
                [[1827.5, 601.0],
                [1117.0, 792.0],
                [1188.5, 1127.5],
                [1189.0, 1129.0],
                [1347.5, 174.0],
                [179.5, 1591.0],
                [804.0, 963.5],
                [1622.0, 1698.0]]
            ),
            {
                'mean_voronoi_area': 805822.2719470514,
                'median_voronoi_area': 337884.2409091698,
                'std_voronoi_area': 1101062.1193302223,
            },
            None,
        ),
        # case with no finite areas found in voronoi
        # regions
        (
            np.array(
                [[1924.5, 841.0],
                [808.5, 485.0],
                [306.5, 538.0],
                [2083.5, 1221.0],
                [344.0, 633.5],
                [724.5, 1047.5]]
            ),
            {},
            "No finite areas found in Voronoi regions",
        ),
    ]
)
def test_calculate_voronoi_stats(caplog, pts, exp, warn_msg):
    """
    test that ``calculate_voronoi_stats`` returns a dictionary of
    key, value pairs from voronoi region calculation OR returns
    an empty dict and warns that no finite areas were found in
    voronoi regions
    """
    caplog.set_level(logging.WARNING)
    actual = da._calculate_voronoi_stats(pts)
    assert actual.keys() == exp.keys()
    for key, value in actual.items():
        assert_allclose(value, exp[key])  
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
            np.array([[850.5, 637.0],
               [511.5, 270.5],
               [308.5, 41.5],]),
            np.array([278.0, 816.0, 671.0]),
            [3, 3, 2.0, 0.0, 1.0, 536.8295523840445,
            1, 1.0, 588.3333333333334],
        ),
        (
        # test case where there ARE NOT enough points to perform
        # triangulation
            "delaunay",
            None,
            None,
            np.array([[1930.0, 1947.0],
                     [167.5, 1710.0]]),
            np.array([475.0, 341.0]),
            [2, 0, 0.0, 0.0, 0.0, np.nan, 2, 0.5, 475.0],
        ),
        # a radius param that only generates a single edge
        (
            "radius",
            100,
            None,
            None,
            None,
            [10, 1, 0.2, 0.42163702135578396,
            0.0, 92.28488500290825, 9, 0.2, 793.0]
        ),
        # a radius param that generates multiple edges
        (
            "radius",
            200,
            None,
            None,
            None,
            [10, 3, 0.6, 0.5163977794943223,
            0.0, 128.97026785384764, 7, 0.2, 156.0]
        ),
        # test case where k=1 nearest neighbors
        (
            "knn", 
            None,
            1,
            None,
            None,
            [10, 6, 1.2, 0.4216370213557839,
            0.0, 217.84017139873353, 4, 0.4, 632.25]
        ),
        # test case where k=3 nearest neighbors (different size output tree)
        (
            "knn", 
            None,
            3,
            None,
            None,
            [10, 20, 4.0, 1.247219128924647,
            0.6066666666666667, 353.41649782593197, 1, 1.0, 510.1]
        ),
        (
        # test case where the number of input points is less than the
        # provided k value, such that the k value is modified to support
        # the appropriate number of neighbors
            "knn",
            None,
            3,
            np.array([[1930.0, 1947.0],
                     [167.5, 1710.0]]),
            np.array([475.0, 341.0]),
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
        areas = rng.integers(1, 1000, 10)
    actual = da._calculate_graph_metrics(
        pts,
        areas,
        method=method,
        k_param=k_param,
        r_param=r_param,
        img_hyp=1e7)
    exp_out = {key: value for key, value in zip(exp_cols, expected)}
    assert actual.keys() == exp_out.keys()
    for key, value in actual.items():
        assert_allclose(value, exp_out[key])
    

@pytest.mark.parametrize("input_df", ["dummy_blobs", None])
def test_extract_blob_properties(make_dummy_blobs, input_df):
    if input_df == "dummy_blobs":
        (df, expected_center_x, expected_center_y,
            expected_areas, expected_radii) = make_dummy_blobs
        expected_centers = np.stack(
            [expected_center_x, expected_center_y], axis=1
        )
    else:
        df = pd.DataFrame(
            columns=[
                "center_x",
                "center_y",
                "area",
                "radius",
                "bbox_xmax",
                "bbox_xmin",
                "bbox_ymin",
                "bbox_ymax",
            ]
        )
        expected_centers = np.array([])
        expected_areas = pd.Series()
        expected_radii = pd.Series()
    actual_centers, actual_areas, actual_radii = da._extract_blob_properties(
        df,
        center_cols=["center_x", "center_y"],
        area_col="area",
        radius_col="radius",
        bbox_cols=["bbox_xmax", "bbox_xmin", "bbox_ymax", "bbox_ymin"],
    )
    assert_allclose(actual_centers, expected_centers)
    if input_df is not None:
        # compare outputs containing actual values
        assert_allclose(actual_areas, expected_areas)
        assert_allclose(actual_radii, expected_radii)
    else:
        # compare outputs containing empty data structures
        assert_series_equal(actual_areas, expected_areas)
        assert_series_equal(actual_radii, expected_radii)

@pytest.mark.parametrize(
    "input_df, n_centroids, exp_nodes, exp_neighbor_dist, exp_mean_nnd, exp_mva",
    [
        ("real_blobs", 5, 5, 307.1795051057866, 194.92048461932964, 796194.8341103308),
        ("real_blobs", 1, 0, np.nan, np.nan, np.nan),
        ("real_blobs", 2, 2, np.nan, 207.663189, np.nan),
        ("real_blobs", 3, 3, 295.04877222436266, 217.1971962781283, np.nan),
        ("real_blobs", 4, 4, 301.3597542865437, 172.05582593652304, 302694.9412916275),
        (None, 0, 0, np.nan, np.nan, np.nan),
    ]
)
def test_calculate_all_spatial_metrics(
    caplog,
    real_blobs,
    input_df,
    n_centroids,
    exp_nodes,
    exp_neighbor_dist,
    exp_mean_nnd,
    exp_mva,
):
    caplog.set_level(logging.WARNING)
    if input_df == "real_blobs":
        df = real_blobs[:n_centroids]
    else:
        df = pd.DataFrame(columns=["center_x", "center_y", "area", "radius", "bbox"])

    actual = da._calculate_all_spatial_metrics(df, graph_method="delaunay", img_shape=[2440, 1115])

    # some sanity checks for outputs
    assert len(actual) == 22
    assert_allclose(actual["mean_blob_area"], df["area"].mean())
    assert actual["graph_num_nodes"] == exp_nodes 
    assert_allclose(actual["graph_avg_neighbor_distance"], exp_neighbor_dist)
    assert_allclose(actual["mean_nnd"], exp_mean_nnd)
    assert_allclose(actual["mean_voronoi_area"], exp_mva)
    # assert that no warnings were logged, indicating that the voronoi stats
    # were not calculated for any array < length 4
    assert caplog.text == ''

@pytest.mark.parametrize(
    "df_empty",
    [
        pd.DataFrame(columns=["center_x", "center_y", "area", "radius", "bbox"]),
        pd.DataFrame(
            {
                "center_x": [10.0, 30.0],
                "center_y": [30.0, 40.0],
                "bbox_xmax": [100, 200],
                "bbox_xmin": [0, 0],
                "bbox_ymax": [100, 150],
                "bbox_ymin": [0, 0],
                # "area" and "radius" intentionally omitted
            }
        ),
    ],
    ids=["empty_df_with_required_cols", "missing_required_cols"],
)
def test_calculate_all_spatial_metrics_else_branch_defaults(df_empty):
    out = da._calculate_all_spatial_metrics(
        df_empty, graph_method="delaunay", img_shape=[100, 100]
    )

    assert out["num_blobs"] == 0
    assert out["total_blob_area"] == 0.0

    for key in (
        "mean_blob_area",
        "median_blob_area",
        "std_blob_area",
        "mean_blob_radius",
        "median_blob_radius",
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
        assert_allclose(out[key], np.nan, equal_nan=True)

    assert_array_equal(
        np.array(
            [
                out["graph_num_nodes"],
                out["graph_num_edges"],
                out["coverage_percentage"],
            ]),
        np.array([0, 0, 0.0]),
    )


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
            "OpenCV",
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
            "BubbleSAM",
        ),
        ("invalid_filename.parquet.gzip", {}, "BubbleSAM"),
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

    expected_df = df.copy()
    out = da._calculate_summary_statistics(
        df,
        group_cols=["Group", "Label"],
        carry_over_cols=["Other"],
        exclude_numeric_cols=None,
        exclude_numeric_regex=[r"^graph_.*"],
    )

    expected = expected_df[["Group", "Label", "Other"]].drop_duplicates().reset_index(drop=True)
    assert_frame_equal(out, expected)

@pytest.mark.parametrize("method, drop_comp_row",
    [
        ("BubbleSAM", None),  # test merge performs correctly on bubblesam data
        ("BubbleSAM", 1),  # test that "left" merge preserves rows without comp data
        ("OpenCV", None),  # test merge performs correctly on opencv data
    ]
)
def test_merge_composition_data(mock_dir, method, drop_comp_row,):
    input_dir, output_dir, comp_csv = mock_dir
    per_img_df = da._process_parquet_files(
        input_dir,
        mode="BubbleSAM",
        graph_method="delaunay",
        img_shape=[10, 10]
    )
    comp_df = pd.read_csv(input_dir / comp_csv)
    if drop_comp_row is not None:
        comp_df.drop(drop_comp_row, inplace=True)
    actual = da._merge_composition_data(
        per_img_df, 
        comp_df, 
        cols_to_add=["Phase_Separation"],
        merge_key="UniqueID"
    )
    # check that the dataframe contains the expected number of rows and columns
    assert actual.shape == (1, 29)

@pytest.mark.parametrize("mode, img_shape, err, err_msg",
    [
        ("BadMode", [10, 10], ValueError, "Mode must be either"),
        ("OpenCV", [10, 10], FileNotFoundError, "No valid files were processed"),
    ]
)
def test_process_parquet_files_errors(
    tmp_path: Path,
    mode: Literal["OpenCV", "BubbleSAM"],
    img_shape: list,
    err: type[Exception],
    err_msg: str,
):
    with pytest.raises(err, match=err_msg):
        da._process_parquet_files(
            tmp_path,
            mode=mode,
            graph_method="delaunay",
            img_shape=img_shape,
        )

@pytest.mark.parametrize(
    "mode, tiff_name, time_label, phase_label, exp_agg_shape, exp_per_shape", 
    [
        (
            "OpenCV",
            (
                'offset -5_bottom_A2_O_Ph_Raw_163c48ec-5ec9'
                '-4b1c-b304-ea40e77f0780_bubble_data.tiff'
            ),
            "1st", None, (1, 94), (1, 29),
        ),
        (
            "BubbleSAM",
            (
                'offset -5_bottom_A1_O_Ph_Raw_b96c0d64-03fd'
                '-4285-824d-e82eafedce90_masks_filtered.tiff'
            ),
            "1st", None, (1, 94), (1, 29),
        ),
        (
            "BubbleSAM",
            (
                'offset -5_bottom_A1_O_Ph_Raw_b96c0d64-03fd'
                '-4285-824d-e82eafedce90_masks_filtered.tiff'
            ),
            None, None, (1, 93), (1, 28)
        ),
        (
            "BubbleSAM",
            (
                'offset -5_bottom_A1_O_Ph_Raw_b96c0d64-03fd'
                '-4285-824d-e82eafedce90_masks_filtered.tiff'
            ),
            "1st", "NaN", (0, 94), (1, 29)
        )
    ]
)
def test_full_analysis_pipeline(
    mock_dir,
    mode,
    tiff_name,
    time_label,
    phase_label,
    exp_agg_shape,
    exp_per_shape,
):
    input_dir, output_dir, comp_csv = mock_dir
    per_image_csv = output_dir / f"per_image_{mode}.csv"
    aggregate_csv = output_dir / f"aggregate_{mode}.csv"
    group_cols = ["Group", "Label", "Time", "Class", "Offset"]
    if time_label is None:
        group_cols.remove("Time") 
    if phase_label == "NaN":
        comp_df = pd.read_csv(comp_csv)
        comp_df["Phase_Separation"] = [np.nan, np.nan]
        comp_df.to_csv(comp_csv, index=False)

    da.full_analysis(
        input_dir=input_dir,
        per_image_csv=per_image_csv,
        aggregate_csv=aggregate_csv,
        mode=mode,
        graph_method="radius",
        r_param=30,
        composition_csv=comp_csv,
        cols_to_add=["Phase_Separation", "Group"],
        time_label=time_label,
        group_cols=group_cols,
        carry_over_cols=["Phase_Separation"],
        img_shape=[90, 97],
    )

    assert per_image_csv.exists()
    assert aggregate_csv.exists()
    df_agg = pd.read_csv(aggregate_csv)
    df_per = pd.read_csv(per_image_csv)
    assert df_per["image_name"].item() == tiff_name
    assert df_agg.shape == exp_agg_shape
    assert df_per.shape == exp_per_shape
    
    assert "Phase_Separation" in df_agg.columns
    time_value = df_per.get("Time")
    assert_array_equal(time_value, time_label)
    assert_array_equal(df_agg.columns[:len(group_cols)], group_cols)
    # numerical checks on the per-image df
    assert_allclose(df_per["std_blob_area"], 107.07261295235324)
    assert_allclose(df_per["graph_degree_std"], 0.9189365834726816)
    assert_allclose(df_per["mean_voronoi_area"], 5450.5100956330925)
    assert_allclose(df_per["coverage_percentage"], 38.04123711340206)
    # numerical checks on the aggregated df
    if phase_label is None:
        assert_allclose(df_agg["mean_blob_area_max"], 332.1)
        assert_allclose(df_agg["graph_avg_degree_min"], 2.2)
        assert_allclose(df_agg["graph_avg_neighbor_distance_min"], 21.13193381222392)
        assert_allclose(df_agg["mean_nnd_median"], 16.695406977528428) 
        assert_allclose(df_agg["mean_voronoi_area_std"], 0.0)

@pytest.mark.parametrize("summary_df_drop_key, comp_df_drop_key, err_msg",
    [
        # missing merge key in summary_df
        (
            "UniqueID", None, "not found in summary_df",
        ),
        # missing specified merge column from composition_df
        (
            None, "Phase", "not found in composition_df",
        ),
    ]
)
def test_merge_composition_data_errors(
    summary_df_drop_key,
    comp_df_drop_key,
    err_msg
):
    """
    Ensure that the appropriate error is raised with expected output
    text when requested cols are missing in composition_df.
    """
    summary_df = pd.DataFrame({"UniqueID": ["id1"], "metric": [42]})
    comp_df = pd.DataFrame({"UniqueID": ["id1"], "Phase": [True]})
    if summary_df_drop_key is not None:
        summary_df.drop(columns=[summary_df_drop_key], inplace=True)
    if comp_df_drop_key is not None:
        comp_df.drop(columns=[comp_df_drop_key], inplace=True)

    with pytest.raises(
        ValueError, match=err_msg
    ):
        da._merge_composition_data(
            summary_df,
            comp_df,
            cols_to_add=["Phase"],
            merge_key="UniqueID",
        )


@pytest.mark.parametrize("file_suff, method",
    [
        ("masks_filtered", "BubbleSAM"),
        ("bubble_data", "OpenCV"),
    ]
)
def test_process_parquet_files_warns_and_continues(
    tmp_path: Path,
    make_dummy_blobs: tuple, 
    file_suff: str,
    method: Literal["OpenCV", "BubbleSAM"],
):
    """
    test that ``process_parquet_files`` warns on unparsable
    but still returns rows for the good files.
    """
    df, _, _, _, _ = make_dummy_blobs 
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # a parquet file with a parseable filename and appropriate contents
    # for returning a dataframe containing a row of calculated metrics
    good = ("offset -1_center_A1_O_Ph_Raw_11111111-"
        f"1111-1111-1111-111111111111_{file_suff}.parquet.gzip")
    df.to_parquet(input_dir / good)
    # a parquet file with appropriate contents but having a filename
    # that is not in the correct format and therefore is not readable
    # by the ``_parse_filename`` function
    unparsable = f"weird_{file_suff}.parquet.gzip"
    df.to_parquet(input_dir / unparsable)

    with pytest.warns(UserWarning) as record:
        df = da._process_parquet_files(
            input_dir,
            mode=method,
            graph_method="delaunay",
            img_shape=[10, 10]
        )

    msgs = [str(w.message) for w in record.list]
    all_msgs = "\n".join(msgs)
    assert "Could not parse metadata from filename" in all_msgs
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 28)


def test_calculate_graph_metrics_bad_method(make_dummy_blobs):
    _, _, pts, areas, _ = make_dummy_blobs
    with pytest.raises(ValueError, match="Invalid input parameter"):
        da._calculate_graph_metrics(pts, areas, method="bad", img_hyp=1e7)


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
            pts, areas, method=method, r_param=r_param, k_param=k_param, img_hyp=1e7,
        )
