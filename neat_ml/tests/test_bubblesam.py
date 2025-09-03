from pathlib import Path
from typing import Tuple, List

import numpy as np
import numpy.testing as npt
import pandas as pd
import cv2
import pytest
import torch
import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images

from neat_ml.bubblesam.bubblesam import (
    show_anns,
    save_masks,
    run_bubblesam,
    process_image,
)
from neat_ml.bubblesam.SAM import SAMModel

# pytestmark = pytest.mark.integration

CHECKPOINT: Path = (
    Path("./neat_ml/sam2/checkpoints/sam2_hiera_large.pt").expanduser().resolve()
)

def _skip_unless_available() -> None:
    """
    Abort the whole module if we cannot load sam2 or the checkpoint.
    """
    pytest.importorskip("sam2", reason="sam2 package is required for SAM-2 tests")
    if not CHECKPOINT.exists():
        pytest.skip(
            f"SAM-2 checkpoint not found at {CHECKPOINT}. "
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
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True

@pytest.fixture(scope="session")
def dummy_rgb(tmp_path_factory) -> Tuple[np.ndarray, Path]:
    """
    Return a plain 100✕100 gray RGB image and its on-disk path.
    """
    img = np.full((100, 100, 3), 200, np.uint8)
    fpath = tmp_path_factory.mktemp("imgs") / "dummy.png"
    cv2.imwrite(str(fpath), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return img, fpath


@pytest.fixture(scope="session")
def baseline_dir() -> Path:
    """
    Folder where baseline PNGs are stored / recorded.
    """
    base = Path(__file__).parent / "baseline"
    base.mkdir(exist_ok=True)
    return base


@pytest.fixture(scope="session")
def real_sam_model() -> SAMModel:
    """
    Actual SAM-2 network on CPU
    """
    model = SAMModel(
        model_config="sam2_hiera_l.yaml",
        checkpoint_path=str(CHECKPOINT),
        device="cpu",
    )
    return model

def _cmp_or_record(expected: Path, actual: Path, *, tol: float = 1e-4) -> None:
    """
    Compare two images pixel-wise using matplotlib's helper.
    If the expected image does not exist, copy actual -> expected
    and mark the test as xfailed.
    """
    if expected.exists():
        diff = compare_images(str(expected), str(actual), tol=tol)
        assert diff is None, f"image diff too large: {diff}"
    else:
        expected.write_bytes(actual.read_bytes())
        pytest.xfail(f"Recorded baseline -> {expected.relative_to(Path.cwd())}")

def test_show_anns_no_error():
    """
    Smoke-test that show_anns draws without raising.
    """
    seg = np.ones((20, 20), bool)
    masks = [{"segmentation": seg, "area": int(seg.sum())}]
    np.random.seed(0)
    fig, ax = plt.subplots()
    plt.sca(ax)
    show_anns(masks)
    plt.close(fig)


def test_save_masks_creates_pngs(tmp_path):
    """
    save_masks() must write one PNG per mask with correct pixel values.
    """
    seg = np.zeros((10, 10), bool)
    seg[2:8, 2:8] = True
    masks = [{"segmentation": seg, "area": int(seg.sum())}]
    save_masks(masks, tmp_path)
    out_file = tmp_path / "mask_0.png"
    assert out_file.exists()
    
    actual = cv2.imread(str(out_file), cv2.IMREAD_GRAYSCALE)
    expected = seg.astype(np.uint8) * 255
    npt.assert_array_equal(actual, expected)

@pytest.fixture(scope="session")
def image_with_circles_fixture(tmp_path_factory) -> Tuple[np.ndarray, Path]:
    """
    Return a 100x100 black RGB image with two white circles and its path.
    """
    img = np.zeros((100, 100, 3), np.uint8)
    white_color = (255, 255, 255)
    filled = -1
    
    cv2.circle(img, center=(30, 30), radius=10, color=white_color, thickness=filled)
    cv2.circle(img, center=(70, 65), radius=15, color=white_color, thickness=filled)

    fpath = tmp_path_factory.mktemp("imgs") / "circles.png"
    cv2.imwrite(str(fpath), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return img, fpath

def test_process_image_generates_pngs_cpu(
    tmp_path, image_with_circles_fixture, real_sam_model, baseline_dir
):
    """
    process_image(debug=True) should run the real model, save two PNGs,
    and match (or record) deterministic baselines.
    """
    np.random.seed(0)
    img_arr, img_fp = image_with_circles_fixture
    out_dir = tmp_path / "run"
    df = process_image(
        image_path=str(img_fp),
        output_dir=str(out_dir),
        sam_model=real_sam_model,
        mask_settings={},
        debug=True,
    )
    assert not df.empty
    
    overlay_png = next(out_dir.glob("*_with_mask.png"))
    contour_png = next(out_dir.glob("*_filtered_contours.png"))

    _cmp_or_record(baseline_dir / overlay_png.name, overlay_png)
    _cmp_or_record(baseline_dir / contour_png.name, contour_png)


def test_sam_internal_api(dummy_rgb, real_sam_model):
    """
    Call _build_model() and generate_masks() directly to confirm they
    execute and return plausible results on CPU.
    """
    torch_model = real_sam_model._build_model()
    assert isinstance(torch_model, torch.nn.Module)
    masks: List[dict] = real_sam_model.generate_masks(
        output_dir=".", image=dummy_rgb[0], mask_settings={}
    )
    npt.assert_array_less(0, len(masks))
    
    seg = masks[0]["segmentation"]
    assert isinstance(seg, np.ndarray)
    npt.assert_equal(seg.dtype, np.dtype("bool"))
    assert "area" in masks[0]
    npt.assert_equal(masks[0]["area"], seg.sum())

def test_run_bubblesam_cpu(tmp_path, image_with_circles_fixture):
    """
    End-to-end test on CPU: run_bubblesam should produce a summary CSV
    and two additional columns.
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
        "use_m2m": True
    }
    df_in = pd.DataFrame({"image_filepath": [str(image_with_circles_fixture[1])]})
    out_dir = tmp_path / "summary_run"
    
    summary = run_bubblesam(
        df_in,
        out_dir,
        model_cfg={"device": "cpu"}, 
        mask_settings=mask_settings,
        debug=False,
    )

    expected_cols = {"image_filepath", "median_radii_SAM", "num_blobs_SAM"}
    assert expected_cols.issubset(summary.columns)
    npt.assert_array_less(0, summary.loc[0, "num_blobs_SAM"])
