import json
from pathlib import Path
from typing import Callable
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import pytest
from matplotlib.testing.compare import compare_images
import os

from neat_ml.utils import lib_plotting as lp

# TODO: enforce style consistency with ``black`` (issue #11)
@pytest.mark.parametrize(
    "writer, fname, binodal_curve",
    [
        (
            lp.titration_diagram,
            "titration_diagram.png",
            False,
        ),
        (
            lp.plot_phase_diagram,
            "phase_diagram_exp.png",
            False,
        ),
        (
            lp.plot_phase_diagram,
            "mathematical_model.png",
            True,
        ),
    ],
)
def test_visual_regression_on_helpers(
    tmp_path: Path,
    baseline_dir: Path,
    synthetic_df: pd.DataFrame,
    writer: Callable,
    fname: str,
    binodal_curve: bool,
):
    extra_kwargs = dict(x_col="Sodium Citrate (wt%)",
        y_col="PEO 8 kg/mol (wt%)",
        phase_col="Phase",
        xrange=[0, 20], yrange=[0, 20])
    
    csv = tmp_path / "input.csv"
    synthetic_df.to_csv(csv, index=False)

    if binodal_curve:
        json_file_path = tmp_path / "test_params.json"
        test_params = {
          "MODEL_A": 0.955,
          "MODEL_B": -5.73,
          "MODEL_C": 581
        }
        with open(json_file_path, 'w') as f:
            json.dump(test_params, f)
        extra_kwargs["json_path"] = str(json_file_path)
        extra_kwargs["binodal_curve"] = True  # type: ignore[assignment]

    out_png = tmp_path / "out" / fname
    writer(file_path=csv, output_path=out_png, **extra_kwargs)
 
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

    out_png = tmp_path / "out" / "plot_two_scatter.png"
    lp.plot_two_scatter(
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
    with open(out_file, 'w') as f:
        json.dump(out_dict, f)
    if json_file == "bad_params.json":
        file_path = os.path.join(tmp_path, json_file)
        err_msg = f"{file_path} {err_msg}"
    with pytest.raises(err_type, match=err_msg):
        lp.load_parameters_from_json(out_file)

@pytest.mark.parametrize("out_path, data_frame, err_msg",
    [
        # Empty directory should raise 'No CSV files found in directory'.
        (
            "empty",
            None, 
            "No CSV files found in directory",
        ),
        # A CSV with only 3 columns triggers the ValueError in the try/except block.
        (
            "bad_csv",
            pd.DataFrame({"A": [1], "B": [2], "C": [3]}),
            "CSV",
        ),
        # A CSV that does not contain the necessary ``Phase`` columns for plotting
        (
            "missing_phase_cols",
            pd.DataFrame({"A": [1], "B": [2], "C": [3], "D": [4], "E": [5]}),
            "Dataframe missing phase columns. Skipping.",
        )
    ],
)
def test_make_phase_diagram_errors(tmp_path: Path,
out_path, data_frame, err_msg):
    """
    test that appropriate errors/warnings are raised in function
    ``make_phase_diagram_figures``
    """
    save_dir = tmp_path / out_path
    save_dir.mkdir()
    phase_cols = ('Phase_Separation_1st', 'Phase_Separation_2nd')

    if out_path in ["bad_csv", "missing_phase_cols"]:
        bad_dir = save_dir / "demo.csv"
        data_frame.to_csv(bad_dir, index=False)
        
    if out_path in ["bad_csv", "empty"]:
        with pytest.raises(ValueError, match=err_msg):
            lp.make_phase_diagram_figures(
                save_dir, 
                tmp_path / "out", 
                phase_cols
            )
    elif out_path == "missing_phase_cols":
        with pytest.warns(UserWarning, match=err_msg):
            lp.make_phase_diagram_figures(
                save_dir, 
                tmp_path / "out", 
                phase_cols
            )

def _build_titration_dir(dir_: Path, df: pd.DataFrame) -> Path:
    """
    Directory with one titration CSV having columns:
    ("Sodium Citrate (wt%)", "PEO 8 kg/mol (wt%)", Phase).
    """
    dir_.mkdir()
    df[["Sodium Citrate (wt%)", "PEO 8 kg/mol (wt%)", "Phase"]].to_csv(
        dir_ / "Synthetic.csv", index=False)
    return dir_


def _build_binodal_dir(dir_: Path, df: pd.DataFrame) -> Path:
    """
    Directory with paired *_Titrate.csv / *_TECAN.csv files.
    """
    dir_.mkdir()
    tit = df[["Sodium Citrate (wt%)", "PEO 8 kg/mol (wt%)"]].iloc[:5]
    tec = df[["Sodium Citrate (wt%)", "PEO 8 kg/mol (wt%)"]].iloc[5:10]
    tit.to_csv(dir_ / "Synthetic_Titrate.csv", index=False)
    tec.to_csv(dir_ / "Synthetic_TECAN_1st.csv", index=False)
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
            "Sodium Citrate (wt%)": df["Sodium Citrate (wt%)"],
            "PEO 8 kg/mol (wt%)": df["PEO 8 kg/mol (wt%)"],
            "Phase_Separation_1st": df["Phase"],
            "Phase_Separation_2nd": df["Phase"],
        }
    )
    out.to_csv(dir_ / "demo.csv", index=False)
    return dir_


def _build_model_csv(path: Path, df: pd.DataFrame) -> Path:
    out = pd.DataFrame(
        {
            "Sodium Citrate (wt%)": df["Sodium Citrate (wt%)"],
            "PEO 8 kg/mol (wt%)": df["PEO 8 kg/mol (wt%)"],
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
    with open(params, 'w') as f:
        json.dump(
            {"MODEL_A": 0.955, "MODEL_B": -5.73, "MODEL_C": 581}, f
        )

    lp.plot_figures(
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
    assert len(pngs) == 5
    
    for png in pngs:
        result = compare_images(
            str(baseline_dir / png.name), 
            str(png), 
            tol=2e-2)
        assert result is None


def test_plot_phase_diagram_error(
    tmp_path: Path,
    baseline_dir: Path,
    synthetic_df: pd.DataFrame,
):
    csv = tmp_path / "input.csv"
    synthetic_df.to_csv(csv, index=False)
    out_png = tmp_path / "out" / "out.png"

    with pytest.raises(ValueError, match="Must provide ``json_path``"): 
        lp.plot_phase_diagram(
            file_path=csv,
            output_path=out_png,
            x_col="Sodium Citrate (wt%)",
            y_col="PEO 8 kg/mol (wt%)",
            phase_col="Phase",
            xrange=[0, 20], yrange=[0, 20],
            binodal_curve = True
        )
