from pathlib import Path
from importlib import resources
import re

import numpy as np
import numpy.testing as npt
from numpy.typing import NDArray
import pandas as pd
import cv2
import pytest
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images

from neat_ml.bubblesam.bubblesam import (
    load_image,
    show_anns,
    save_masks,
    run_bubblesam,
    process_image,
    analyze_and_filter_masks,
)
from neat_ml.bubblesam.SAM import SAMModel

CHECKPOINT: Path = Path("./neat_ml/sam2/checkpoints/sam2_hiera_large.pt")

def _skip_unless_available(model_chkpt: Path = CHECKPOINT) -> None:
    """
    Abort the whole module if we cannot load sam2 or the checkpoint.
    """
    pytest.importorskip("sam2", reason="sam2 package is required for SAM-2 tests")
    if not CHECKPOINT.exists():
        pytest.skip(
            f"SAM-2 checkpoint not found at {model_chkpt}. "
            "Install it to run integration tests.",
            allow_module_level=True,
        )

_skip_unless_available()

def test_setup_cuda_does_not_crash_on_cpu(monkeypatch):
    """
    Ensures that calling setup_cuda() in an environment with no GPU
    completes without error.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = SAMModel(
        model_config="sam2_hiera_l.yaml",
        checkpoint_path=str(CHECKPOINT),
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
def test_setup_cuda_on_real_gpu():
    """
    Verifies that setup_cuda() correctly configures torch backends on
    a live GPU. This test only runs if a CUDA device is found.
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = SAMModel(
        model_config="sam2_hiera_l.yaml",
        checkpoint_path=str(CHECKPOINT),
        device="cuda",
    )
    model.setup_cuda()
    if torch.cuda.get_device_properties(0).major >= 8:
        assert torch.backends.cuda.matmul.allow_tf32
        assert torch.backends.cudnn.allow_tf32

@pytest.fixture(scope="module")
def real_sam_model(model_chkpt: Path = CHECKPOINT) -> SAMModel:
    """
    Actual SAM-2 network on CPU
    """
    return SAMModel(
        model_config="sam2_hiera_l.yaml",
        checkpoint_path=str(model_chkpt),
        device="cpu",
    )

def test_load_image_missing_file_raises(tmp_path: Path) -> None:
    missing_path = tmp_path / "does_not_exist.png"
    expected = re.escape(f"Image at path {missing_path} not found.")

    with pytest.raises(FileNotFoundError, match=expected):
        load_image(str(missing_path))


@pytest.mark.parametrize("slice_idx, fn_call",
    [
        (slice(0, 0), "empty_anns"),
        (slice(None), "imshow"),   
    ],
)
def test_show_anns(mocker, slice_idx, fn_call):
    """
    Smoke-test that show_anns draws without raising.
    """
    mock_fig = mocker.MagicMock()
    mock_ax = mocker.MagicMock()
    mocker.patch("matplotlib.pyplot.gca", return_value=mock_ax)
    seg = np.ones((20, 20), bool)
    masks = [{"segmentation": seg, "area": int(seg.sum())}]
    show_anns(masks[slice_idx])
    if fn_call == "imshow":
        mock_ax.imshow.assert_called_once()
    if fn_call == "empty_anns":
        mock_ax.imshow.assert_not_called()


def test_save_masks_creates_pngs(tmp_path: Path):
    """
    save_masks() must write one PNG per mask with correct pixel values.
    """
    seg = np.zeros((10, 10), bool)
    seg[2:8, 2:8] = True
    masks = [{"segmentation": seg, "area": int(seg.sum())}]
    save_masks(masks, str(tmp_path))
    out_file = tmp_path / "mask_0.png"

    actual = cv2.imread(str(out_file), cv2.IMREAD_GRAYSCALE)
    assert actual is not None
    expected = seg.astype(np.uint8) * 255
    npt.assert_array_equal(actual, expected)

@pytest.fixture(scope="module")
def image_with_circles_fixture(tmp_path_factory) -> Path:
    """
    Return a path to a 100x100 black RGB image with two white circles.
    """
    img = np.zeros((100, 100, 3), np.uint8)
    white, filled = (255, 255, 255), -1
    cv2.circle(img, center=(30, 30), radius=10, color=white, thickness=filled)
    cv2.circle(img, center=(70, 65), radius=15, color=white, thickness=filled)
    fpath = tmp_path_factory.mktemp("imgs") / "circles.png"
    cv2.imwrite(str(fpath), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return fpath

def test_process_image_generates_pngs_cpu(
    tmp_path: Path, image_with_circles_fixture: Path, real_sam_model: SAMModel
):
    """
    process_image(debug=True) should run the real model, save two PNGs,
    and match (or record) deterministic baselines.
    """
    np.random.seed(0)
    img_fp = image_with_circles_fixture
    stem = Path(img_fp).stem
    out_dir = tmp_path / "run"

    df = process_image(
        image_path=str(img_fp),
        output_dir=str(out_dir),
        sam_model=real_sam_model,
        mask_settings={},
        debug=True,
    )
    assert not df.empty

    actual_overlay  = out_dir / f"{stem}_with_mask.png"
    actual_contours = out_dir / f"{stem}_filtered_contours.png"

    with resources.as_file(resources.files(__package__) / "baseline") as baseline_dir:
        desired_overlay  = baseline_dir / actual_overlay.name
        desired_contours = baseline_dir / actual_contours.name

        result1 = compare_images(str(desired_overlay),  str(actual_overlay),  tol=1e-4)
        result2 = compare_images(str(desired_contours), str(actual_contours), tol=1e-4)
        assert result1 is None and result2 is None

def test_sam_internal_api(real_sam_model: SAMModel):
    """
    Call _build_model() and generate_masks() directly to confirm they
    execute and return plausible results on CPU.
    """
    torch_model = real_sam_model._build_model()
    assert isinstance(torch_model, torch.nn.Module)

    dummy_rgb = np.full((100, 100, 3), 200, np.uint8)
    masks = real_sam_model.generate_masks(
        output_dir=".", image=dummy_rgb, mask_settings={}
    )
    assert len(masks) > 0

    seg = masks[0]["segmentation"]
    assert isinstance(seg, np.ndarray)
    assert seg.dtype is np.dtype("bool")
    assert "area" in masks[0]
    assert masks[0]["area"] ==  seg.sum()

def test_run_bubblesam_cpu(tmp_path: Path, image_with_circles_fixture: Path):
    """
    End-to-end test on CPU: run_bubblesam should produce a summary CSV
    with expected columns.
    """
    mask_settings = {
        "points_per_side": 8,
        "points_per_batch": 16,
        "pred_iou_thresh": 0.80,
        "stability_score_thresh": 0.80,
        "stability_score_offset": 0.1,
        "crop_n_layers": 2,
        "box_nms_thresh": 0.1,
        "crop_n_points_downscale_factor": 1,
        "min_mask_region_area": 5,
        "use_m2m": True,
    }
    model_cfg = {
        "model_config": "sam2_hiera_l.yaml",
        "checkpoint_path": "./neat_ml/sam2/checkpoints/sam2_hiera_large.pt",
        "device": "cpu",
    }
    df_in = pd.DataFrame({"image_filepath": [str(image_with_circles_fixture)]})
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
    assert summary["median_radii_SAM"].item() == 12.778613837669742
    assert "circles.png" in summary["image_filepath"].item()


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
