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
from importlib.resources import files
from importlib.resources.abc import Traversable
import logging

from neat_ml.bubblesam.bubblesam import (
    show_anns,
    save_masks,
    run_bubblesam,
    bubblesam_detection,
    analyze_and_filter_masks,
)
from neat_ml.bubblesam.SAM import SAMModel

CHECKPOINT = files("neat_ml.sam2").joinpath("checkpoints/sam2_hiera_tiny.pt")

def _skip_unless_available(model_chkpt: Traversable = CHECKPOINT) -> None:
    """
    Abort the whole module if we cannot load sam2 or the checkpoint.
    """
    pytest.importorskip("neat_ml.sam2", reason="sam2 package is required for SAM-2 tests")
    if not model_chkpt.is_file():
        pytest.skip(
            f"SAM-2 checkpoint not found at {model_chkpt}. "
            "Install it to run integration tests.",
            allow_module_level=True,
        )

_skip_unless_available()

@pytest.mark.skipif(
    (torch.cuda.is_available() or torch.backends.mps.is_available()),
    reason="This test is intended for systems without GPU support"
)
def test_setup_cuda_does_not_crash_on_cpu(
    model_chkpt: Traversable = CHECKPOINT,
):
    """
    Ensures that calling setup_cuda() in an environment with no GPU
    completes without error.
    """
    model = SAMModel(
        model_config="sam2_hiera_t.yaml",
        checkpoint_path=model_chkpt,
        device="cpu",
    )
    try:
        model.setup_cuda()
    except Exception as e:
        pytest.fail(f"setup_cuda() raised an unexpected exception on CPU: {e}")

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="This test requires a CUDA-enabled GPU"
)
def test_setup_cuda_on_real_gpu(model_chkpt: Traversable = CHECKPOINT):
    """
    Verifies that setup_cuda() correctly configures torch backends on
    a live GPU. This test only runs if a CUDA device is found.
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = SAMModel(
        model_config="sam2_hiera_t.yaml",
        checkpoint_path=model_chkpt,
        device="cuda",
    )
    model.setup_cuda()
    if torch.cuda.get_device_properties(0).major >= 8:
        assert torch.backends.cuda.matmul.allow_tf32
        assert torch.backends.cudnn.allow_tf32

@pytest.fixture(scope="module")
def real_sam_model(model_chkpt: Traversable = CHECKPOINT) -> SAMModel:
    """
    Actual SAM-2 network on CPU
    """
    return SAMModel(
        model_config="sam2_hiera_t.yaml",
        checkpoint_path=model_chkpt,
        device="cpu",
    )

def test_bubblesam_detection_missing_file_raises(
    tmp_path: Path,
    real_sam_model: SAMModel,
) -> None:
    missing_path = tmp_path / "does_not_exist.png"
    expected = f"Image at path {missing_path} not found."
    
    rng = np.random.default_rng()
    with pytest.raises(FileNotFoundError, match=expected):
        bubblesam_detection(missing_path, tmp_path, real_sam_model, {}, rng)


@pytest.mark.parametrize("slice_idx, return_call",
    [
        (slice(0, 0), 0),
        (slice(None), 1),   
    ],
)
def test_show_anns(slice_idx, return_call):
    """
    Smoke-test that show_anns draws without raising.
    """
    rng = np.random.default_rng(0)
    seg = np.ones((20, 20), bool)
    fig, ax = plt.subplots(1, 1)
    masks = [{"segmentation": seg, "area": seg.sum()}]
    show_anns(masks[slice_idx], ax, rng)
    assert len(ax.get_images()) == return_call


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

def test_bubblesam_detection_generates_pngs_cpu(
    tmp_path: Path,
    image_with_circles_fixture: Path,
    real_sam_model: SAMModel,
    mask_settings: dict,
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

    df = bubblesam_detection(
        image_path=img_fp,
        output_dir=out_dir,
        sam_model=real_sam_model,
        mask_settings=mask_settings,
        rng=rng,
        debug=True,
    )
    assert not df.empty

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

def test_sam_internal_api(real_sam_model: SAMModel):
    """
    Call _build_model() and generate_masks() directly to confirm they
    execute and return plausible results on CPU.
    """
    torch_model = real_sam_model._build_model()
    assert isinstance(torch_model, torch.nn.Module)

    dummy_rgb = np.full((100, 100, 3), 200, np.uint8)
    masks = real_sam_model.generate_masks(
        image=dummy_rgb, mask_settings={}
    )
    assert len(masks) > 0

    seg = masks[0]["segmentation"]
    assert isinstance(seg, np.ndarray)
    assert seg.dtype is np.dtype("bool")
    assert "area" in masks[0]
    assert masks[0]["area"] == seg.sum()

def test_run_bubblesam_cpu(
    tmp_path: Path,
    image_with_circles_fixture: Path,
    mask_settings,
    model_chkpt: Traversable = CHECKPOINT,
):
    """
    End-to-end test on CPU: run_bubblesam should produce a summary CSV
    with expected columns.
    """
    model_cfg = {
        "model_config": "sam2_hiera_t.yaml",
        "checkpoint_path": model_chkpt,
        "device": "cpu",
    }
    df_in = pd.DataFrame({"image_filepath": [image_with_circles_fixture]})
    out_dir = tmp_path / "summary_run"

    summary = run_bubblesam(
        df_in,
        out_dir,
        model_cfg=model_cfg,
        mask_settings=mask_settings,
        debug=False,
    )

    expected_cols = {"image_filepath", "median_radii_SAM", "num_blobs_SAM"}
    assert expected_cols.issubset(set(summary.columns))
    assert summary["num_blobs_SAM"].item() == 2
    assert_allclose(summary["median_radii_SAM"].item(), 12.778613837669742)
    assert summary["image_filepath"].item().name == "circles.png"


@pytest.mark.parametrize("center_pixel",
    [   
        0, 1            
    ]
)
def test_analyze_and_filter_masks_no_props(center_pixel):
    """
    test that the function returns an empty dataframe when either
    the mask has no ROI's or the detected area has zero perimeter
    """
    image = np.zeros([3, 3])
    image[1, 1] = center_pixel
    mask_df = pd.DataFrame({"segmentation": [image.astype(bool)]}) 
    out_df = analyze_and_filter_masks(mask_df)    
    assert out_df.empty


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="This test is intended for macos systems with torch mps support"
)
def test_setup_cuda_mps_warns(
    caplog,
    model_chkpt = CHECKPOINT,
):
    """
    test that a warning is raised when initializing ``setup_cuda``
    with `mps` backend
    """
    caplog.set_level(logging.WARNING) 
    model = SAMModel(
        model_config="sam2_hiera_t.yaml",
        checkpoint_path=model_chkpt,
        device="mps",
    )
    model.setup_cuda()
    assert "Support for MPS devices is preliminary" in caplog.text
