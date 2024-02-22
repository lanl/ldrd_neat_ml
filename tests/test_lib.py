from neat_ml import lib


def test_cesar_cg_rdf_labels():
    # from: https://gitlab.lanl.gov/treddy/ldrd_neat_ml/-/issues/25
    # ensure Cesar's CG-MD RDF data columns have names
    # that are more appropriate than unlabelled floating point
    # values
    actual_df_cols = lib.read_in_cesar_cg_md_data().columns
    for col in actual_df_cols[10:]:
        assert str(col).startswith("RDF_")
