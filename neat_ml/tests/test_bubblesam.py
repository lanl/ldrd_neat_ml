from pathlib import Path
from importlib import resources

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
import cv2
import pytest
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images
import logging
from pandas.testing import assert_frame_equal

from neat_ml.bubblesam.bubblesam import (
    show_anns,
    save_masks,
    run_bubblesam,
    bubblesam_detection,
    analyze_and_filter_masks,
)
from neat_ml.bubblesam.SAM import SAMModel


@pytest.fixture(scope="module")
def real_sam_model(mask_settings) -> SAMModel:
    """
    Actual SAM-2 network on CPU
    """
    return SAMModel(
        mask_settings=mask_settings,
        checkpoint_path="facebook/sam2.1-hiera-tiny",
        device="cpu",
    )

def test_bubblesam_detection_missing_file_raises(
    tmp_path: Path,
    real_sam_model: SAMModel,
):
    missing_path = tmp_path / "does_not_exist.png"
    expected = f"Image at path {missing_path} not found."
    
    rng = np.random.default_rng()
    with pytest.raises(FileNotFoundError, match=expected):
        bubblesam_detection(
            missing_path,
            tmp_path,
            real_sam_model,
            rng,
            0.0, 0.0,
        )

@pytest.mark.parametrize("slice_idx, return_call",
    [
        (slice(0, 0), 0),
        (slice(None), 1),   
    ],
)
def test_show_anns(slice_idx, return_call):
    """
    Smoke-test that show_anns draws without raising.
    Verifies that annotations are rendered in descending area order (largest first).
    """
    rng = np.random.default_rng(0)
    seg_large = np.ones((20, 20), bool)
    seg_small = np.zeros((20, 20), bool)
    seg_small[0:5, 0:5] = True

    fig, ax = plt.subplots(1, 1)

    # pass small mask before large to test sorting
    masks = [
        {"segmentation": seg_small, "area": seg_small.sum()},
        {"segmentation": seg_large, "area": seg_large.sum()},
    ]

    show_anns(masks[slice_idx], ax, rng)
    assert len(ax.get_images()) == return_call

    if return_call:
        rendered = ax.get_images()[0].get_array()
        # the large mask covers the entire canvas, so if sorting is correct,
        # the small mask pixels (top-left 5x5) must be rendered on top of
        # the large mask, meaning they should differ from the rest of the image.
        center_pixel = rendered[10, 10]   # inside large only
        overlap_pixel = rendered[2, 2]    # inside both small (on top) and large

        # if sorted correctly, small is drawn after large, overlap differs from center
        assert not np.allclose(overlap_pixel, center_pixel)


def test_save_masks_creates_pngs(tmp_path: Path):
    """
    save_masks() must write one PNG per mask with correct pixel values.
    """
    seg = np.zeros((10, 10), bool)
    seg[2:8, 2:8] = True
    masks = [{"segmentation": seg, "area": seg.sum()}]
    save_masks(masks, tmp_path)
    out_file = tmp_path / "mask_0.png"

    actual = cv2.imread(out_file, cv2.IMREAD_GRAYSCALE)  # type: ignore[call-overload]
    expected = seg.astype(np.uint8) * 255
    assert_array_equal(actual, expected)

@pytest.mark.parametrize("device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="only run on cuda enabled gpus"
                )
            ]
        ),
        pytest.param(
            "mps",
            marks=[
                pytest.mark.skipif(
                    not torch.backends.mps.is_available(), reason="only run on mps enabled gpus"
                )
            ]
        ),
    ]
)
def test_bubblesam_detection_generates_pngs(
    tmp_path: Path,
    image_with_circles_fixture: Path,
    real_sam_model: SAMModel,
    mask_settings: dict,
    device: str,
):
    """
    bubblesam_detection(debug=True) should run the real model, save two PNGs,
    and match (or record) deterministic baselines.
    """
    rng = np.random.default_rng(0)
    img_fp = image_with_circles_fixture
    stem = img_fp.stem
    out_dir = tmp_path / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    real_sam_model.device = device

    df = bubblesam_detection(
        image_path=img_fp,
        output_dir=out_dir,
        sam_model=real_sam_model,
        rng=rng,
        area_threshold=25.0,
        circularity_threshold=0.90,
        debug=True,
    )
    saved_df = pd.read_parquet(
        out_dir / "circles_masks_filtered.parquet.gzip",
        engine="fastparquet",
    )
    saved_df["bbox"] = saved_df["bbox"].apply(tuple)
    saved_df['contour'] = saved_df['contour'].apply(
        lambda x: [np.array(arr, dtype='int32') for arr in x]
    )
    assert_frame_equal(df, saved_df)

    actual_overlay  = out_dir / f"{stem}_with_mask.png"
    actual_contours = out_dir / f"{stem}_filtered_contours.png"

    with resources.as_file(resources.files(__package__) / "baseline") as baseline_dir:
        desired_overlay  = baseline_dir / actual_overlay.name
        desired_contours = baseline_dir / actual_contours.name

        result1 = compare_images(
            desired_overlay, actual_overlay,  tol=1e-4
        )  # type: ignore[call-overload]
        # loosened tolerance below to accommodate cross-platform testing
        result2 = compare_images(
            desired_contours, actual_contours, tol=1e-2
        )  # type: ignore[call-overload]
        assert result1 is None
        assert result2 is None

@pytest.mark.parametrize("device",
    [
        "cpu",
        pytest.param(
            "cuda", 
            marks=[
                pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="only run on cuda enabled gpus"
                )
            ]
        ),
        pytest.param(
            "mps",
            marks=[
                pytest.mark.skipif(
                    not torch.backends.mps.is_available(), reason="only run on macos with mps backend"
                )
            ]
        ),
    ]
)
def test_sam_internal_api(
    real_sam_model: SAMModel,
    reference_images: tuple, 
    tmp_path: Path, 
    device,
):
    """
    Call generate_masks() directly to confirm execution
    and return plausible results across devices.
    """
    real_sam_model.device = device
    input_img = cv2.imread(reference_images[3])  #type: ignore[call-overload]
    top_corner = input_img[:256, :256, :]  #type: ignore[index]
    masks = real_sam_model.generate_masks(image=top_corner)
    iou = np.mean([x.get("predicted_iou") for x in masks])  #type: ignore[arg-type]
    total_area = np.sum([x.get("area") for x in masks])
    assert len(masks) == 24
    assert total_area == 5091
    assert_allclose(iou, 0.880697)

@pytest.mark.parametrize("device",
    [
        "cpu",
        pytest.param(
            "gpu",
            marks=[
                pytest.mark.skipif(
                    not torch.backends.mps.is_available()
                    and not torch.cuda.is_available(), reason="only run when gpu available"
                )
            ]
        ),
    ]
)
def test_run_bubblesam(
    tmp_path: Path,
    image_with_circles_fixture: Path,
    mask_settings,
    device,
):
    """
    End-to-end test of run_bubblesam. Output dataframe should contain
    appropriate columns and reproducable values associated with the
    detection of two circles in a binary image.
    """
    
    detection_cfg = {
        "model_cfg":
            {
                "mask_settings": mask_settings,
                "checkpoint_path": "facebook/sam2.1-hiera-tiny",
                "device": device,
            }
    }
    df_in = pd.DataFrame({"image_filepath": [image_with_circles_fixture]})
    out_dir = tmp_path / "summary_run"

    summary = run_bubblesam(
        df_in,
        out_dir,
        detection_cfg=detection_cfg,
        debug=False,
    )

    expected_cols = {"image_filepath", "median_radii_SAM", "num_blobs_SAM"}
    assert expected_cols.issubset(set(summary.columns))
    assert summary["num_blobs_SAM"].item() == 2
    assert_allclose(summary["median_radii_SAM"].item(), 12.778613837669742)
    assert summary["image_filepath"].item().name == "circles.tiff"


@pytest.mark.parametrize("center_pixel",
    [   
        0, 1            
    ]
)
def test_analyze_and_filter_masks_no_props(center_pixel):
    """
    test that the function returns an empty dataframe when either
    the mask has no regions of interest (ROI) or the detected area has zero perimeter
    """
    image = np.zeros([3, 3])
    image[1, 1] = center_pixel
    mask_df = pd.DataFrame({"segmentation": [image.astype(bool)]}) 
    out_df = analyze_and_filter_masks(mask_df, 25.0, 0.90, "cpu")
    assert out_df.empty


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="This test is intended for macos systems with torch mps support"
)
def test_setup_cuda_mps_warns(
    caplog,
    mask_settings,
):
    """
    test that a warning is raised when initializing ``setup_cuda``
    with `mps` backend
    """
    caplog.set_level(logging.WARNING) 
    SAMModel(
        mask_settings=mask_settings,
        checkpoint_path="facebook/sam2.1-hiera-tiny",
        device="gpu",
    )
    assert "Support for MPS devices is preliminary" in caplog.text


def test_run_bubblesam_model_cfg_error():
    """
    test that ``run_bubblesam`` raises value error
    when provided empty `model_cfg` input
    """
    with pytest.raises(ValueError, match="Must provide model configuration"):
        run_bubblesam(pd.DataFrame(), Path("output"), detection_cfg={})

@pytest.mark.parametrize("seg_params, exp_bbox",
    [  
        # a test case where the segmentation contains two disjoint areas
        ([[50, 60], [40, 45]], (50, 50, 60, 60)),
        # a test case where the segmentation contains a region that touches
        # the image boundary at the bottom right corner
        ([[90, 100]], (90, 90, 100, 100)),
    ]
)
def test_bubblesam_contours(seg_params, exp_bbox):
    """
    test that running `analyze_and_filter_masks` generates a dataframe with
    only a single contour per detection and without background areas
    """
    # create two segmentation maps, one that takes up the whole image (background)
    # and one containing the segmentation map generated using the test case parameters
    seg = np.ones((100, 100)).astype(bool)
    seg2 = np.zeros((100, 100)).astype(bool)
    for seg_param in seg_params:
        start = seg_param[0]
        end = seg_param[1]
        seg2[start:end, start:end] = True
    input_df = pd.DataFrame({"segmentation": [seg, seg2]})
    # call `analyze_and_filter_masks` to return filtered dataframe
    # (the circularity of a perfect square is ~0.8, so lower the
    # circularity threshold so that the background only gets filtered
    # out by the bounding box area)
    df = analyze_and_filter_masks(input_df, 25, 0.7, device="cpu")
    # assert that there is only a single dataframe row after filtration
    # corresponding to the appropriate segmentation map to keep from `seg2`
    assert df.bbox.item() == exp_bbox
    assert df.contour.item().shape == (36, 2)
