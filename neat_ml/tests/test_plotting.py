import json
from pathlib import Path
from importlib import resources
from typing import Callable, Generator, Any
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.testing.compare import compare_images
import os

from neat_ml.utils import lib_plotting as pmf
from neat_ml.utils import figure_utils

# TODO: enforce style consistency with ``black`` (issue #11)
@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """Deterministic (seeded) composition/phase dataframe."""
    rng = np.random.default_rng(7)
    x = rng.uniform(0.0, 20.0, 30)
    y = rng.uniform(0.0, 20.0, 30)
    phase = (x + y > 20.0).astype(int)
    return pd.DataFrame({"X": x, "Y": y, "Phase": phase})

@pytest.fixture(scope="session")
def baseline_dir() -> Generator[Any, Any, Any]:
    """
    Directory that stores the reference (golden) images.
    """
    ref = resources.files("neat_ml.tests") / "baseline"
    with resources.as_file(ref) as path:
        yield path

def test_plot_gmm_decision_regions_visual_and_logic(
    tmp_path: Path,
    baseline_dir: Path,
    synthetic_df: pd.DataFrame,
):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    gmm, labels, boundary = figure_utils.plot_gmm_decision_regions(
        df=synthetic_df,
        x_col="X",
        y_col="Y",
        phase_col="Phase",
        ax=ax,
        xrange=[0, 20],
        yrange=[0, 20],
        n_components=2,
        random_state=42,
        boundary_color="red",
        resolution=200,
        decision_alpha=1,
        plot_regions=True,
        region_colors=["aquamarine", "lightsteelblue"],
    )
    assert labels.shape == (len(synthetic_df),)
    assert boundary is None or boundary.shape[1] == 2
    assert hasattr(gmm, "predict")

    out_png = tmp_path / "gmm_decision_regions.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    
    # for ``compare_images`` tests ``tol`` was loosened from 1e-4 to
    # 2e-2 to accommodate cross-platform testing as discussed at
    # https://github.com/lanl/ldrd_neat_ml/pull/1#issuecomment-3463967764
    result = compare_images(
        str(baseline_dir / "gmm_decision_regions.png"), 
        str(out_png), 
        tol=2e-2)
    assert result is None

def test_plot_gmm_composition_phase_visual_and_logic(
    tmp_path: Path,
    baseline_dir: Path,    
    synthetic_df: pd.DataFrame,
):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    figure_utils.plot_gmm_composition_phase(
        df=synthetic_df,
        x_col="X",
        y_col="Y",
        phase_col="Phase",
        ax=ax,
        point_cmap=["#FFFFCC", "dodgerblue"],
    )
    scatters = [c for c in ax.collections if np.asarray(c.get_offsets()).size > 0]
    assert len(scatters) == 2

    out_png = tmp_path / "gmm_composition_phase.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    result = compare_images(
        str(baseline_dir / "gmm_composition_phase.png"), 
        str(out_png), 
        tol=2e-2)
    assert result is None

@pytest.mark.parametrize(
    "writer, fname, extra_kwargs",
    [
        (
            pmf.titration_diagram,
            "titration_diagram.png",
            dict(x_col="X", y_col="Y", phase_col="Phase", xrange=[0, 20], yrange=[0, 20]),
        ),
        (
            pmf.phase_diagram_exp,
            "phase_diagram_exp.png",
            dict(x_col="X", y_col="Y", phase_col="Phase", xrange=[0, 20], yrange=[0, 20]),
        ),
        (
            pmf.mathematical_model,
            "mathematical_model.png",
            dict(x_col="X", y_col="Y", phase_col="Phase", xrange=[0, 20], yrange=[0, 20]),
        ),
    ],
)
def test_visual_regression_on_helpers(
    tmp_path: Path,
    baseline_dir: Path,
    synthetic_df: pd.DataFrame,
    writer: Callable,
    fname: str,
    extra_kwargs: dict,
):
    
    csv = tmp_path / "input.csv"
    synthetic_df.to_csv(csv, index=False)

    if writer.__name__ == "mathematical_model":
        json_file_path = tmp_path / "test_params.json"
        test_params = {
          "MODEL_A": 0.955,
          "MODEL_B": -5.73,
          "MODEL_C": 581
        }
        with open(json_file_path, 'w') as f:
            json.dump(test_params, f)
        extra_kwargs["json_path"] = str(json_file_path)

    out_png = tmp_path / fname
    writer(file_path=csv, output_path=out_png, **extra_kwargs)

    assert out_png.is_file()
    arr = plt.imread(out_png)
    assert np.ptp(arr[..., :3]) > 0.0
    
    result = compare_images(
        str(baseline_dir / fname), 
        str(out_png), 
        tol=2e-2)
    assert result is None

def test_plot_two_scatter_visual_regression(
    tmp_path: Path,
    baseline_dir: Path,
):
    csv1 = tmp_path / "scatter_A.csv"
    pd.DataFrame({"X": [1, 2, 3], "Y": [3, 2, 1]}).to_csv(csv1, index=False)

    csv2 = tmp_path / "scatter_B.csv"
    pd.DataFrame({"X": [1, 2, 3], "Y": [1, 2, 3]}).to_csv(csv2, index=False)

    out_png = tmp_path / "plot_two_scatter.png"
    pmf.plot_two_scatter(
        csv1_path=csv1,
        csv2_path=csv2,
        output_path=out_png,
        xlim=[0, 4],
        ylim=[0, 4],
    )
    result = compare_images(
        str(baseline_dir/ "plot_two_scatter.png"), 
        str(out_png), 
        tol=2e-2)
    assert result is None

@pytest.mark.parametrize("json_file, out_dict, err_msg, err_type",
[
    (
	"bad_params.json",
	{"MODEL_A": 0.1, "MODEL_B": 0.0},
	"is missing required keys: {\'MODEL_C\'}",
	KeyError,
    ),
    (
	"bad_value_type.json",
	{"MODEL_A": 10.5, "MODEL_B": "not-a-number", "MODEL_C": 20.0},
	"Parameter 'MODEL_B' must be a number, but got str.",
        TypeError,
    ),
]
)
def test_load_json_errors(tmp_path: Path,
json_file, out_dict, err_msg, err_type):
    """
    The JSON is valid but omits MODEL_C -> KeyError expected.
    
    AND	

    Tests that a TypeError is raised if a parameter value is not a number.
    
    The JSON file contains all required keys, but the value for 'MODEL_B'
    is a string, which should trigger a TypeError.
    """
    out_file = tmp_path / json_file
    json.dump(out_dict, out_file.open("w"))
    if json_file == "bad_params.json":
        file_path = os.path.join(tmp_path, json_file)
        err_msg = f"{file_path} {err_msg}"
    with pytest.raises(err_type, match=err_msg):
        pmf.load_parameters_from_json(out_file)

@pytest.mark.parametrize("out_path, out_file, err_msg",
    [
        (
            "emtpy",
            None, 
            "No CSV files found in directory"
        ),
        (
            "bad_csv",
            pd.DataFrame({"A": [1], "B": [2], "C": [3]}),
            "CSV",
        ),
    ],
)
def test_make_phase_diagram_errors(tmp_path: Path,
out_path, out_file, err_msg):
    """
    Empty directory should raise 'No CSV files found in directory'.
    
    AND     

    A CSV with only 3 columns triggers the ValueError in the try/except block.
    """
    save_dir = tmp_path / out_path
    save_dir.mkdir()
    phase_cols=('Phase_Separation_1st', 'Phase_Separation_2nd')

    if out_path == "bad_csv":
        bad_dir = save_dir / "demo.csv"
        out_file.to_csv(bad_dir, index=False)
    
    with pytest.raises(ValueError, match=err_msg):
        pmf.make_phase_diagram_figures(
            save_dir, 
            tmp_path / "out", 
            phase_cols
        )


def _build_titration_dir(dir_: Path, df: pd.DataFrame) -> Path:
    """
    Directory with one titration CSV (X,Y,Phase).
    """
    dir_.mkdir()
    df[["X", "Y", "Phase"]].to_csv(dir_ / "Synthetic.csv", index=False)
    return dir_


def _build_binodal_dir(dir_: Path, df: pd.DataFrame) -> Path:
    """
    Directory with paired *_Titrate.csv / *_TECAN.csv files.
    """
    dir_.mkdir()
    tit = df[["X", "Y"]].iloc[:5]
    tec = df[["X", "Y"]].iloc[5:10]
    tit.to_csv(dir_ / "Synthetic_Titrate.csv", index=False)
    tec.to_csv(dir_ / "Synthetic_TECAN.csv", index=False)
    return dir_


def _build_phase_dir(dir_: Path, df: pd.DataFrame) -> Path:
    """
    Directory with one CSV suitable for make_phase_diagram_figures.
    """
    dir_.mkdir()
    out = pd.DataFrame(
        {
            "Junk1": 0,
            "Junk2": 0,
            "Junk3": 0,
            "X": df["X"],
            "Y": df["Y"],
            "Phase_Separation_1st": df["Phase"],
            "Phase_Separation_2nd": df["Phase"],
        }
    )
    out.to_csv(dir_ / "demo.csv", index=False)
    return dir_


def _build_model_csv(path: Path, df: pd.DataFrame) -> Path:
    out = pd.DataFrame(
        {
            "Sodium Citrate (wt%)": df["X"],
            "PEO 8 kg/mol (wt%)": df["Y"],
            "Phase_Separation_2nd": df["Phase"],
        }
    )
    out.to_csv(path, index=False)
    return path

def test_wrappers_and_pipeline(
    synthetic_df: pd.DataFrame,
    baseline_dir: Path,
    tmp_path: Path,
):
    work = tmp_path / "work"
    work.mkdir()
    out_dir = work / "out"
    out_dir.mkdir()
    phase_cols=('Phase_Separation_1st', 'Phase_Separation_2nd')

    tit_dir = _build_titration_dir(work / "tit_dir", synthetic_df)
    bin_dir = _build_binodal_dir(work / "bin_dir", synthetic_df)
    phase_dir = _build_phase_dir(work / "phase_dir", synthetic_df)
    model_csv = _build_model_csv(work / "model.csv", synthetic_df)
    model_png = work / 'model.png'

    params = work / "params.json"
    json.dump(
        {"MODEL_A": 0.955, "MODEL_B": -5.73, "MODEL_C": 581},
        params.open("w"),
    )

    pmf.plot_figures(
        titration_csv_dir=tit_dir,
        binodal_csv_dir=bin_dir,
        csv_phase_dir=phase_dir,
        out_dir=out_dir,
        mat_model_csv=model_csv,
        mat_model_png=model_png,
        json_path=params,
        phase_cols=phase_cols,
        xrange=[0, 20],
        yrange=[0, 20],
    )

    pngs = sorted(out_dir.glob("*.png"))
    assert pngs, "The main pipeline produced no PNGs"

    for png in pngs:
        result = compare_images(
            str(baseline_dir / png.name), 
            str(png), 
            tol=2e-2)
        assert result is None

