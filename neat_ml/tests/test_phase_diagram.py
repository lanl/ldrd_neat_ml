from pathlib import Path
import matplotlib
matplotlib.use("Agg")                     

from matplotlib.testing.compare import compare_images
import numpy as np
import pandas as pd
import pytest

from neat_ml.phase_diagram.plot_phase_diagram import construct_phase_diagram

@pytest.fixture(scope="module")
def sample_phase_df():
    """
    Generates a synthetic DataFrame with two distinct clusters of points,
    representing two phases, to create a less complex and more realistic
    phase diagram for visual testing.
    """
    rng = np.random.default_rng(seed=0)
    n_per_phase = 25

    mean0 = [3, 2]
    cov0 = [[3, 0.5], [0.5, 2]]
    coords0 = rng.multivariate_normal(mean0, cov0, n_per_phase)
    df0 = pd.DataFrame(coords0, columns=["Dextran", "PEO"])
    df0["TruePhase"] = 0

    mean1 = [12, 6]
    cov1 = [[4, -1], [-1, 3]]
    coords1 = rng.multivariate_normal(mean1, cov1, n_per_phase)
    df1 = pd.DataFrame(coords1, columns=["Dextran", "PEO"])
    df1["TruePhase"] = 1
    df = pd.concat([df0, df1], ignore_index=True)
    df[["Dextran", "PEO"]] = df[["Dextran", "PEO"]].clip(lower=0)
    df = df.sample(frac=1, random_state=rng).reset_index(drop=True)
    flip = rng.random(len(df)) < 0.1
    df["PredPhase"] = np.where(flip, 1 - df["TruePhase"], df["TruePhase"])

    return df

def test_construct_phase_diagram_image(tmp_path, sample_phase_df):
    """
    Tests the visual output of construct_phase_diagram by comparing it
    to a baseline image.
    """
    baseline_dir = Path(__file__).parent / "baseline"
    baseline_png = baseline_dir / "test_phase_diagram.png"
    out_png = tmp_path / "phase.png"

    construct_phase_diagram(
        sample_phase_df,
        dex_col="Dextran",
        peo_col="PEO",
        true_phase_col="TruePhase",
        pred_phase_col="PredPhase",
        title="Synthetic Phase Diagram",
        out_dir=str(tmp_path),
        fname="phase",
    )

    if not baseline_png.exists():
        pytest.fail(f"Baseline image did not exist. It has been created at {baseline_png}. Please review it and run the test again.")

    rms = compare_images(str(baseline_png), str(out_png), tol=10.0)
    assert rms is None, f"Phase diagram image differs from baseline (RMS={rms})"