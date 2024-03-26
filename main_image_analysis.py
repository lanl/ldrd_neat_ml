"""
Given an image captured for a % PEO / % DEX
mixture, try to automatically determine
the (average?) diameter of droplets present
in the image, as a proxy for phase separation.

Presumably, we'll use a diameter of `0` to indicate
absence of relevant droplets.

Part of the purpose here is to avoid human bias
in the generation of the response variables for
our downstream ML work.
"""

from neat_ml import lib

import glob

import numpy as np


def main():
    # Of course the image data is too large to
    # commit to the repo, so it is downloaded from Google Drive
    # by the user independently, before running this code
    data_root_path = "/Users/treddy/LANL/LDRD_DR_NEAT_data/Images"

    # let's find all the % PEO / % DEX .tiff filepaths and do
    # a few sanity checks
    img_filepaths = glob.glob(f"{data_root_path}/**/*.tiff",
                              recursive=True)
    assert len(img_filepaths) == 46

    # check that images all have the same pixel dims
    # (2456 x 2052 at the time of writing)
    lib.check_image_dim_consistency(img_filepaths)

    # Parse out the (WT % DEX, WT % PEO) information
    # from the image filepaths, and start building relevant
    # information into a DataFrame
    df = lib.build_df_from_exp_img_paths(img_filepaths)
    assert df.shape == (46, 3)
    # had a bug where PEO/DEX columns were accidentally
    # the same, so rule that out:
    diff = df["WT% PEO"] - df["WT% DEX"]
    assert not np.allclose(diff, np.zeros(df.shape[0]))

    # Produce a standard plot of the binary phase system
    # points (we don't have phase "labels" yet)
    # TODO: should rename this function to something more generic
    lib.plot_input_data_cesar_MD(df=df,
                                 title="Plate Reader Image Data for PEO/DEX\n",
                                 fig_name="plate_reader_image_points_",
                                 )

    # there are a variety of ways we could try to estimate
    # the droplet sizes (diameters); perhaps it makes sense to try a few
    # and compare them

    # 1) Using the Hough Transform

    df = lib.skimage_hough_transform(df=df, debug=True)
    lib.plot_input_data_cesar_MD(df=df,
                                 title="Plate Reader Image Data for PEO/DEX\n",
                                 fig_name="plate_reader_image_points_hough_",
                                 title_addition="(labels from median Hough radii)",
                                 y_pred=df["median_radii_skimage_hough"],
                                 cbar_label="median Hough radii",
                                 )

    # 2) Using Blob Detection Techniques
    df = lib.blob_detection(df=df, debug=True)
    lib.plot_input_data_cesar_MD(df=df,
                                 title="Plate Reader Image Data for PEO/DEX\n",
                                 fig_name="plate_reader_image_points_DoH_sigma",
                                 title_addition="(labels from median DoH sigma/radii)",
                                 y_pred=df["median_radii_DoH"],
                                 cbar_label="median DoH sigma",
                                 )
    lib.plot_input_data_cesar_MD(df=df,
                                 title="Plate Reader Image Data for PEO/DEX\n",
                                 fig_name="plate_reader_image_points_DoH_num_blobs",
                                 title_addition="(labels from DoH num blobs)",
                                 y_pred=df["num_blobs_DoH"],
                                 norm="symlog",
                                 cbar_label="symlog scaled blob count",
                                 )




if __name__ == "__main__":
    main()
