from neat_ml import lib

import pytest


def test_cesar_cg_rdf_labels():
    # from: https://gitlab.lanl.gov/treddy/ldrd_neat_ml/-/issues/25
    # ensure Cesar's CG-MD RDF data columns have names
    # that are more appropriate than unlabelled floating point
    # values
    actual_df_cols = lib.read_in_cesar_cg_md_data().columns
    for col in actual_df_cols[10:]:
        assert str(col).startswith("CG_RDF_")


@pytest.mark.parametrize("df, expected_prefix", [
    (lib.read_in_cesar_cg_md_data(), "CG_"),
    (lib.read_in_cesar_all_atom_md_data(), "AA_"),
    ],
)
def test_cesar_data_column_prefix_by_type(df, expected_prefix):
    # ensure that the CG and AA MD data column
    # titles are prefixed with a string that
    # clearly distinguished the two types of simulations
    # (helpful for i.e., feature importance analysis)
    for col_name in df.columns[3:]:
        assert str(col_name).startswith(expected_prefix)
