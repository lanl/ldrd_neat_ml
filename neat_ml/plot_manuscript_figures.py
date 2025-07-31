import os
from pathlib import Path
from typing import Optional, TypedDict, cast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import json

from . import figure_utils

PHASE_COLS = ("Phase_Separation_1st", "Phase_Separation_2nd")
FIG_3_CSV = Path("neat_ml/data/figure_data/Titration_Figures/")
FIG_6_CSV = Path("neat_ml/data/figure_data/Binodal_Comparison_Figures/")
CSV_PHASE_DIR = Path("neat_ml/data/Binary_Mixture_Phase_Information")
OUT_DIR = Path("neat_ml/data/Figures_for_Manuscript")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MAT_MODEL_CSV = Path(
    "neat_ml/data/Binary_Mixture_Phase_Information/PEO8K_Sodium_Citrate_Composition_Phase.csv"
)
JSON_PATH = Path(
    "neat_ml/data/mathematical_model_parameters.json"
)
MAT_MODEL_PNG = OUT_DIR / (
    "PEO8K_Sodium_Citrate_Phase_Diagram_Experiment_Literature_Comparison.png"
)

class ModelVars(TypedDict):
    MODEL_A: float
    MODEL_B: float
    MODEL_C: float
    MODEL_R2: float

def load_parameters_from_json(
    path: str
) -> ModelVars:
    """
    Read and validate the model fit parameters file.
    
    Parameter
    ---------
    path: str
        Path to the .json file
    
    Returns
    -------
    ModelVars
        A dictionary-like object containing the validated model parameters.
        This object is cast to the ModelVars type for static analysis.
    """
    
    with open(path, "r") as fh:
        raw = json.load(fh)

    missing = ModelVars.__required_keys__ - raw.keys()
    if missing:
        raise KeyError(f"{path} is missing required keys: {missing}")

    return cast(ModelVars, raw)

def titration_diagram(
    file_path: str,
    x_col: str,
    y_col: str,
    phase_col: str,
    xrange: list[int],
    yrange: list[int],
    output_path: str,
) -> None:
    """
    Load data and plot two-phase scatter diagram.

    Parameters
    ----------
    file_path : str
        Path to the input CSV file.
    x_col : str
        Name of the column for x-axis values.
    y_col : str
        Name of the column for y-axis values.
    phase_col : str
        Column name for phase labels (0 or 1).
    xrange : list[int]
        [min, max] range for the x-axis.
    yrange : list[int]
        [min, max] range for the y-axis.
    output_path : str
        Path to save the output PNG image.

    Returns
    -------
    None
        Displays the plot and writes image to output_path.
    """
    df = pd.read_csv(file_path)

    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)

    single = df[df[phase_col] == 0]
    ax.scatter(
        single[x_col],
        single[y_col],
        c="dodgerblue",
        marker="s",
        s=120,
        edgecolors="black",
        linewidths=1,
        label="Single Phase",
    )

    two = df[df[phase_col] == 1]
    ax.scatter(
        two[x_col],
        two[y_col],
        c="#FFFFCC",
        marker="^",
        s=120,
        edgecolors="black",
        linewidths=1,
        label="Two Phase",
    )

    ax.set_xlabel(x_col, fontsize=24, fontweight="bold")
    ax.set_ylabel(y_col, fontsize=24, fontweight="bold")
    figure_utils._set_axis_style(ax, xrange, yrange)

    ax.legend(loc="upper right", framealpha=0.8, fontsize=14)

    dest_dir = os.path.dirname(output_path)
    if dest_dir and not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)

    fig.savefig(
        output_path, 
        bbox_inches="tight",
        pad_inches=1.0
    )
    plt.close(fig) 
    print(f"Plot saved successfully to {output_path}")

def plot_two_scatter(
    csv1_path: str,
    csv2_path: str,
    output_path: str,
    xlim: Optional[list[int]] = None,
    ylim: Optional[list[int]] = None,
) -> None:
    """
    Read two Excel sheets and plot styled scatter comparison.

    Parameters
    ----------
    csv1_path : str
        Path to Titration.CSV file
    csv2_path : str
        Path to TECAN.CSV file
    output_path : str
        Path to save the output image.
    xlim : Optional[list[float]]
        [min, max] range for x-axis.
    ylim : Optional[list[float]]
        [min, max] range for y-axis.

    Returns
    -------
    None
        Displays the plot and writes image to output_path.
    """

    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)

    x_col = list(df2.columns)[0]
    y_col = list(df2.columns)[1]

    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)

    ax.scatter(
        df1[x_col],
        df1[y_col],
        c="purple",
        marker="^",
        s=120,
        edgecolors="black",
        linewidths=1,
        label="Titration",
    )
    ax.scatter(
        df2[x_col],
        df2[y_col],
        c="yellow",
        marker="^",
        s=120,
        edgecolors="black",
        linewidths=1,
        label="Our Method",
    )

    ax.set_xlabel(x_col, fontsize=24, fontweight="bold")
    ax.set_ylabel(y_col, fontsize=24, fontweight="bold")
    figure_utils._set_axis_style(ax, xlim, ylim)

    ax.legend(loc="upper right", framealpha=0.8, fontsize=14)

    dest_dir = os.path.dirname(output_path)
    if dest_dir and not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
    
    fig.savefig(
        output_path, 
        bbox_inches="tight",
        pad_inches=1.0
    )
    plt.close(fig)
    print(f"Plot saved successfully to {output_path}")

def mathematical_model(
    file_path: str,
    json_path: str,
    x_col: str,
    y_col: str,
    phase_col: str,
    output_path: str,
    xrange: list[int] = [0, 20],
    yrange: list[int] = [0, 20],
):
    """
    Generate and save a phase diagram plot from composition data.

    Parameters
    ----------
    file_path : str
        Path to the CSV input file.
    json_path: str
        Path to the .json file for model parameters
    x_col : str
        Column name for x-axis values.
    y_col : str
        Column name for y-axis values.
    phase_col : str
        Column name for phase labels.
    xrange : list[int]
        Two-element list [min, max] for x-axis range.
    yrange : list[int]
        Two-element list [min, max] for y-axis range.
    output_path : str
        Path to save the output plot image.

    Returns
    -------
    None
        Displays the plot and writes the image to output_path.
    """
    df = pd.read_csv(file_path)

    model_vars = load_parameters_from_json(json_path)

    A: float = model_vars["MODEL_A"]
    B: float = model_vars["MODEL_B"]
    C: float = model_vars["MODEL_C"]
    R2: float = model_vars["MODEL_R2"]

    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)

    figure_utils.plot_gmm_decision_regions(
        df,
        x_col,
        y_col,
        phase_col,
        ax=ax,
        xrange=xrange,
        yrange=yrange,
        region_colors=["aquamarine", "lightsteelblue"],
        decision_alpha=1,
    )

    figure_utils.plot_gmm_composition_phase(
        df,
        x_col,
        y_col,
        phase_col,
        ax=ax,
        point_cmap=["#FFFFCC", "dodgerblue"],
    )
    
    raw_x = np.linspace(xrange[0], xrange[1], 500, dtype=float)
    frac_x = raw_x / 100.0
    model_y = A * np.exp(B * np.sqrt(frac_x) - C * frac_x**3) * 100.0

    ax.plot(
        raw_x,
        model_y,
        color="black",
        linewidth=2.5,
        label=f"Model fit: a={A:.3f}, b={B:.2f}, c={C:.0f}",
    )

    handles = [
        Patch(facecolor="aquamarine", edgecolor="black",
              label="Two-Phase (Experiment)"),
        Patch(facecolor="lightsteelblue", edgecolor="black",
              label="Single-Phase (Experiment)"),
        Line2D([0], [0], marker="^", color="w",
               markerfacecolor="#FFFFCC", markeredgecolor="black",
               markersize=10, label="Two-Phase (Experiment)"),
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor="dodgerblue", markeredgecolor="black",
               markersize=10, label="Single-Phase (Experiment)"),
        Line2D([0], [0], color="red", lw=3, label="Decision Boundary"),
        Line2D([0], [0], color="black", lw=2.5, label="Model fit"),
    ]

    ax.legend(
        handles=handles, 
        bbox_to_anchor=(1.02, 1),
        loc="upper left", 
        framealpha=0.8, 
        fontsize=11
    )

    ax.text(
        0.5,
        0.9,
        f"R$^2$ = {R2:.4f}",
        transform=ax.transAxes,
        ha="center",
        fontsize=14,
    )

    ax.set_xlabel(x_col, fontsize=24, fontweight="bold")
    ax.set_ylabel(y_col, fontsize=24, fontweight="bold")
    figure_utils._set_axis_style(ax, xrange, yrange)

    dest_dir = os.path.dirname(output_path)
    if dest_dir and not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)

    fig.savefig(
        output_path, 
        bbox_inches="tight",
        pad_inches=1.0
    )
    plt.close(fig)
    print(f"Plot saved successfully to {output_path}")

def phase_diagram_exp(
    file_path: str,
    x_col: str,
    y_col: str,
    phase_col: str,
    xrange: list[int],
    yrange: list[int],
    output_path: str,
) -> None:
    """
    Load data, generate the phase diagram, and write the image.

    Parameters
    ----------
    file_path : str
        Path to a CSV file with the required columns.
    x_col, y_col : str
        Column names for the two composition axes.
    phase_col : str
        Column holding the phase indicator used to fit the GMM.
    xrange, yrange : list[int]
        Two-element lists specifying axis extents.
    output_path : str
        Destination filename (PNG, SVG, etc.) for the saved figure.

    Returns
    -------
        None
    """

    df = pd.read_csv(file_path)
    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)

    figure_utils.plot_gmm_decision_regions(
        df,
        x_col,
        y_col,
        phase_col,
        ax,
        xrange=xrange,
        yrange=yrange,
        region_colors=["aquamarine", "lightsteelblue"],
        decision_alpha=1,
    )
    figure_utils.plot_gmm_composition_phase(
        df,
        x_col,
        y_col,
        phase_col,
        ax,
        point_cmap=["#FFFFCC", "dodgerblue"],
    )
    
    handles = [
        Patch(facecolor="aquamarine", edgecolor="black",
              label="Two-Phase (Experiment)"),
        Patch(facecolor="lightsteelblue", edgecolor="black",
              label="Single-Phase (Experiment)"),
        Line2D([0], [0], marker="^", color="w",
               markerfacecolor="#FFFFCC", markeredgecolor="black",
               markersize=10, label="Two-Phase (Experiment)"),
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor="dodgerblue", markeredgecolor="black",
               markersize=10, label="Single-Phase (Experiment)"),
        Line2D([0], [0], color="red", lw=3, label="Decision Boundary"),
    ]
    ax.legend(
        handles=handles, 
        bbox_to_anchor=(1.02, 1),
        loc="upper left", 
        framealpha=0.8, 
        fontsize=11
    )

    ax.set_xlabel(x_col, fontsize=24, fontweight="bold")
    ax.set_ylabel(y_col, fontsize=24, fontweight="bold")
    figure_utils._set_axis_style(ax, xrange, yrange)

    dest_dir = os.path.dirname(output_path)
    if dest_dir and not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)

    fig.savefig(
        output_path, 
        bbox_inches="tight",
        pad_inches=1.0
    )
    plt.close(fig)
    print(f"Plot saved to '{output_path}'.")

def make_titration_figures(
    csv_dir: Path,
    out_dir: Path
) -> None:
    """
    Processes a CSV file to generate titration plots from each sheet.

    This function generates titration phase diagram as a PNG image.

    The first row of each CSV file is expected to contain the titles for the
    x-axis, y-axis, and phase columns, respectively. The resulting
    PNG files are named after the CSV file and are saved in the
    specified output directory.

    Parameters
    ----------
    csv_dir : Path
        The file path to the input csv files containing the titration data.
    out_dir : Path
        The path to the directory where the output CSV files and PNG
        images will be saved.

    Returns
    -------
    None
        This function does not return any value; its purpose is to create
        and save files to the disk.
    """

    for csv_path in sorted(csv_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        x_title, y_title, phase_title = df.columns[:3]
        x_rng = [0, int(df[x_title].max()) + 1]
        y_rng = [0, int(df[y_title].max()) + 1]

        png_path = out_dir / f"{csv_path.stem}_Titration_Phase_Diagram.png"

        titration_diagram(
            file_path=str(csv_path),
            x_col=x_title,
            y_col=y_title,
            phase_col=phase_title,
            xrange=x_rng,
            yrange=y_rng,
            output_path=str(png_path),
        )

def make_binodal_comparison_figures(
    csv_dir: Path,
    out_dir: Path
) -> None:
    """
    Generates Titrate-vs-TECAN comparison plots for every paired dataset.

    This function scans an Excel workbook for sheets with names matching a
    specific pattern (e.g., "Titrate_A", "TECAN_A"). It groups these sheets
    by their common identifier (e.g., "A").

    For each identified pair, it reads the data, assuming the first row
    contains the x and y titles. The data from each sheet is saved to a
    separate CSV file. Finally, it generates a scatter plot comparing the
    two datasets and saves it as a PNG image.

    Parameters
    ----------
    csv_dir : Path
        The file path to the input CSV file directory.
    out_dir : Path
        The path to the directory where the output CSV files and PNG
        images will be saved.

    Returns
    -------
    None
        This function does not return a value; it saves files to disk.

    Raises
    ------
    AssertionError
        If any dataset ID is missing its corresponding "Titrate" or "TECAN"
        sheet.
    """

    titrate: dict[str, Path] = {}
    tecan:   dict[str, Path] = {}

    for p in csv_dir.glob("*_Titrate.csv"):
        titrate[p.stem[:-8]] = p       # strip '_Titrate'
    for p in csv_dir.glob("*_TECAN.csv"):
        tecan[p.stem[:-6]] = p         # strip '_TECAN'

    for ds_id in sorted(titrate):
        csv_tit = titrate[ds_id]
        csv_tec = tecan[ds_id]

        df_tit = pd.read_csv(csv_tit)
        df_tec = pd.read_csv(csv_tec)
        x_col, y_col = df_tit.columns[:2]
        x_rng, y_rng = figure_utils._axis_ranges(df_tit, df_tec, x_col, y_col)

        png_path = out_dir / f"Figure_6_{ds_id}_Binodal_Comparison.png"

        plot_two_scatter(
            csv1_path=str(csv_tit),
            csv2_path=str(csv_tec),
            output_path=str(png_path),
            xlim=x_rng,
            ylim=y_rng,
        )

def make_phase_diagram_figures(
    csv_dir: Path, 
    out_dir: Path
) -> None:
    """Iterates through CSV files to produce two diagrams per file.

    This function searches the specified csv_dir for all files ending
    in .csv. For each file found, it reads the data and attempts to
    generate up to two phase diagrams.

    It assumes that the x and y data are in the 4th and 5th columns,
    respectively. It then checks for the existence of specific phase
    columns (defined in PHASE_COLS) to create a diagram for each
    phase found.

    Parameters
    ----------
    csv_dir : Path
        The path to the directory containing the input CSV files.
    out_dir : Path
        The path to the directory where the output PNG images will be saved.

    Returns
    -------
    None
        This function does not return a value; it saves files to disk.
    """
    csv_files = sorted(p for p in csv_dir.iterdir() if p.suffix.lower() == ".csv")
    if not csv_files:
        raise ValueError(f"No CSV files found in directory: {csv_dir}")

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)

        try:
            x_title, y_title = df.columns[3:5]
        except ValueError as exc:
            raise ValueError(
                f"CSV {csv_path} must have at least five columns; got {len(df.columns)}"
            ) from exc

        x_rng = [0, int(df[x_title].max()) + 1]
        y_rng = [0, int(df[y_title].max()) + 1]
        ds_id = csv_path.stem

        for tag, phase_col in zip(("1st", "2nd"), PHASE_COLS):
            if phase_col not in df.columns:
                continue

            png_path = out_dir / f"{ds_id}_Phase_Diagram_{tag}.png"
            phase_diagram_exp(
                file_path=str(csv_path),
                x_col=x_title,
                y_col=y_title,
                phase_col=phase_col,
                xrange=x_rng,
                yrange=y_rng,
                output_path=str(png_path),
            )

def plot_figures(
    titration_csv_dir: Path = FIG_3_CSV,
    binodal_csv_dir:   Path = FIG_6_CSV,
    csv_phase_dir: Path = CSV_PHASE_DIR,
    out_dir: Path = OUT_DIR,
    mat_model_csv: Path = MAT_MODEL_CSV,
    json_path: Path = JSON_PATH,
    xrange: list[int] = [0, 21],
    yrange: list[int] = [0, 38]
) -> None:
    """
    Run all three figure-generation pipelines.

    The routine is a thin orchestrator: it delegates the heavy
    lifting to the specialised helper functions defined in
    this module (make_titration_figures, make_binodal_comparison_figures,
    make_phase_diagram_figures, and mathematical_model).
    Each parameter has a sensible module-level default so that

    Parameters
    ----------
    titration_csv_dir : pathlib.Path, default FIG_3_CSV
        CSV files containing individual titration datasets
    binodal_csv_dir : pathlib.Path, default FIG_6_CSV
        CSV files containing paired “Titrate/TECAN” binodal data
    csv_phase_dir : pathlib.Path, default CSV_PHASE_DIR
        Directory containing one or more CSV files with composition/phase data
    out_dir : pathlib.Path, default OUT_DIR
        Destination directory for all generated artefacts (PNGs and any
        intermediate CSV exports).
    mat_model_csv : pathlib.Path, default MAT_MODEL_CSV
        Single CSV sheet that feeds the mathematical model comparison plot
    json_path : pathlib.Path, default JSON_PATH
        JSON file containing the calibrated model parameters
        MODEL_A, MODEL_B, MODEL_C and MODEL_R2.

    Returns
    -------
    None
        The function is executed for its side-effects: it writes a collection
        of PNG files to out_dir.  No value is returned.
    """
    make_titration_figures(titration_csv_dir, out_dir)
    make_binodal_comparison_figures(binodal_csv_dir, out_dir)
    make_phase_diagram_figures(csv_phase_dir, out_dir)

    model_png = out_dir / MAT_MODEL_PNG.name

    mathematical_model(
        file_path=str(mat_model_csv),
        json_path=str(json_path),
        x_col="Sodium Citrate (wt%)",
        y_col="PEO 8 kg/mol (wt%)",
        phase_col="Phase_Separation_2nd",
        xrange=xrange,
        yrange=yrange,
        output_path=str(model_png),
    )


if __name__ == "__main__":
    plot_figures()
