import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import json

from . import figure_utils

def load_parameters_from_json(path: str) -> dict[str, float]:
    """
    Read and validate the model fit parameters file.

    The parameters were obtained from equation 1 of the
    following literature:

    Silvério, Sara C., et al. "Effect of aqueous two-phase 
    system constituents in different poly (ethylene 
    glycol)-salt phase diagrams." Journal of Chemical & 
    Engineering Data 57.4 (2012): 1203-1208.

    Returns
    -------
    dict[str, float]
        A dictionary containing the validated model parameters.
    """
    REQUIRED_KEYS = {"MODEL_A", "MODEL_B", "MODEL_C"}
    
    with open(path, "r") as fh:
        params = json.load(fh)

    missing_keys = REQUIRED_KEYS - params.keys()
    if missing_keys:
        raise KeyError(f"{path} is missing required keys: {missing_keys}")

    for key, value in params.items():
        if not isinstance(value, (int, float)):
            raise TypeError(f"Parameter '{key}' must be a number, but got {type(value).__name__}.")

    return params

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
        Plots and writes image to output_path.
    """
    df = pd.read_csv(file_path)

    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)

    scatter_cfg: list[dict[str, Any]] = [
        {"val": 0, "color": "dodgerblue", "marker": "s", "label": "Single Phase"},
        {"val": 1, "color": "#FFFFCC", "marker": "^", "label": "Two Phase"},
    ]

    for cfg in scatter_cfg:
        mask: pd.Series = df[phase_col] == cfg["val"]
        ax.scatter(
            df.loc[mask, x_col],
            df.loc[mask, y_col],
            c=cfg["color"],
            marker=cfg["marker"],
            s=180,
            edgecolors="black",
            linewidths=1,
            label=cfg["label"],
        )

    ax.set_xlabel(x_col, fontsize=36, fontweight="bold")
    ax.set_ylabel(y_col, fontsize=36, fontweight="bold")
    figure_utils._set_axis_style(ax, xrange, yrange)

    ax.legend(loc="upper right", framealpha=0.8, fontsize=30)

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
    xlim: Optional[list[int]]=None,
    ylim: Optional[list[int]]=None,
) -> None:
    """
    Read two CSV files and plot styled scatter comparison.

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
        Plots and writes image to output_path.
    """

    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)

    x_col = list(df2.columns)[0]
    y_col = list(df2.columns)[1]

    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)

    scatter_cfg:  list[Dict[str, Any]] = [
        {"df": df1, "color": "purple", "label": "Titration"},
        {"df": df2, "color": "yellow", "label": "Our Method"},
    ]

    for cfg in scatter_cfg:
        data: pd.DataFrame = cfg["df"]
        ax.scatter(
            data[x_col],
            data[y_col],
            c=cfg["color"],
            marker="^",
            s=120,
            edgecolors="black",
            linewidths=1,
            label=cfg["label"],
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
    xrange: list[int],
    yrange: list[int],
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
        Plots and writes the image to output_path.
    """
    df = pd.read_csv(file_path)

    model_vars = load_parameters_from_json(json_path)

    A: float = model_vars["MODEL_A"]
    B: float = model_vars["MODEL_B"]
    C: float = model_vars["MODEL_C"]

    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)

    figure_utils.plot_gmm_decision_regions(
        df,
        x_col,
        y_col,
        phase_col,
        ax=ax,
        xrange=xrange,
        yrange=yrange,
        n_components=2,
        random_state=42,
        boundary_color="red",
        resolution=200,
        decision_alpha=1,
        plot_regions=True,
        region_colors=["aquamarine", "lightsteelblue"],
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
        n_components=2,
        random_state=42,
        boundary_color="red",
        resolution=200,
        decision_alpha=1,
        plot_regions=True,
        region_colors=["aquamarine", "lightsteelblue"],
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

    This function scans CSV files in the directory with names matching a
    specific pattern (e.g., "A_Titrate", "A_TECAN"). It groups these CSV
    by their common identifier (e.g., "A").

    For each identified pair, it reads the data, assuming the first row
    contains the x and y titles. Finally, it generates a scatter plot 
    comparing the two datasets and saves it as a PNG image.

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
        x_rng, y_rng = figure_utils._axis_ranges(df_tit, df_tec, x_col, y_col, pad=2)

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
    out_dir: Path,
    phase_cols: Tuple[str, str]
) -> None:
    """Iterates through CSV files to produce two diagrams per file.

    This function searches the specified csv_dir for all files ending
    in .csv. For each file found, it reads the data and attempts to
    generate up to two phase diagrams.

    It assumes that the x and y data are in the 4th and 5th columns,
    respectively. It then checks for the existence of specific phase
    columns (defined in phase_cols) to create a diagram for each
    phase found.

    Parameters
    ----------
    csv_dir : Path
        The path to the directory containing the input CSV files.
    out_dir : Path
        The path to the directory where the output PNG images will be saved.
    phase_cols : Tuple[str, str]
        The tuple of phase separation column names used for phase diagram

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

        for tag, phase_col in zip(("1st", "2nd"), phase_cols):
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
    titration_csv_dir: Path,
    binodal_csv_dir:   Path,
    csv_phase_dir: Path,
    out_dir: Path,
    mat_model_csv: Path,
    mat_model_png: Path,
    json_path: Path,
    phase_cols: Tuple[str, str],
    xrange: list[int],
    yrange: list[int]
) -> None:
    """
    Run all three figure-generation pipelines.

    The routine is a thin orchestrator: it delegates the heavy
    lifting to the specialised helper functions defined in
    this module (make_titration_figures, make_binodal_comparison_figures,
    make_phase_diagram_figures, and mathematical_model).

    Parameters
    ----------
    titration_csv_dir : pathlib.Path
        CSV files containing individual titration datasets
    binodal_csv_dir : pathlib.Path
        CSV files containing paired “Titrate/TECAN” binodal data
    csv_phase_dir : pathlib.Path
        Directory containing one or more CSV files with composition/phase data
    out_dir : pathlib.Path
        Destination directory for all generated artefacts (PNGs and any
        intermediate CSV exports).
    mat_model_csv : pathlib.Path
        Single CSV sheet that feeds the mathematical model comparison plot
    mat_model_png : pathlib.Path
        Output filename for the mathematical model phase diagram
    json_path : pathlib.Path, default JSON_PATH
        JSON file containing the calibrated model parameters
        MODEL_A, MODEL_B, and MODEL_C.
    phase_cols: Tuple[str, str]
        Tuple of strings to represent the phase conlumn names 
    xrange: list[int]
        X-axis range for the plot
    yrange: list[int]
        Y-axis range for the plot

    Returns
    -------
    None
        The function is executed for its side-effects: it writes a collection
        of PNG files to out_dir.  No value is returned.
    """


    make_titration_figures(titration_csv_dir, out_dir)
    make_binodal_comparison_figures(binodal_csv_dir, out_dir)
    make_phase_diagram_figures(csv_phase_dir, out_dir, phase_cols)

    mathematical_model(
        file_path=str(mat_model_csv),
        json_path=str(json_path),
        x_col="Sodium Citrate (wt%)",
        y_col="PEO 8 kg/mol (wt%)",
        phase_col="Phase_Separation_2nd",
        xrange=xrange,
        yrange=yrange,
        output_path=str(mat_model_png),
    )