import warnings
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from neat_ml.utils.utils import (
    _axis_ranges,
    _set_axis_style,
    plot_gmm_decision_regions,
    plot_gmm_composition_phase,
)

warnings.filterwarnings("ignore", category=UserWarning)

__all__: Sequence[str] = ["construct_phase_diagram"]

def _figure_4x3(
    width_px: int,
    height_px: int,
    dpi: int,
):
    """
    Returns a 4:3 canvas (e.g., 1200x900 px) with constrained layout.
    """
    fig, ax = plt.subplots(
        figsize=(width_px / dpi, height_px / dpi),
        dpi=dpi,
        constrained_layout=True
    )
    return fig, ax

def _format_ticks(
    ax: plt.Axes,
    tick_fontsize: int,
    tick_weight: str
) -> None:
    ax.tick_params(axis="both", which="both", labelsize=tick_fontsize)
    for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lbl.set_fontweight(tick_weight)

def _apply_aspect(
    ax: plt.Axes,
    equal_aspect: bool = False
) -> None:
    """
    If equal_aspect=False, ensure no prior 'equal' or 'box aspect' sticks.
    If True, enforce equal data scaling (square axes within the 4:3 canvas).
    """
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    else:
        ax.set_box_aspect(None)
        ax.set_aspect("auto", adjustable="datalim")

def construct_phase_diagram(
    df: pd.DataFrame,
    *,
    dex_col: str,
    peo_col: str,
    true_phase_col: str,
    pred_phase_col: str,
    title: str,
    out_dir: Path = Path("phase_plots"),
    fname: str = "diagram",
) -> None:
    """
    Make a side-by-side experimental vs. model phase diagram.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing composition and phase information.
    dex_col : str
        Column name for dextran (x-axis values).
    peo_col : str
        Column name for PEO (y-axis values).
    true_phase_col : str
        Column name containing experimental phase labels (0=two-phase, 1=single).
    pred_phase_col : str
        Column name containing model-predicted phase labels.
    title : str
        Title for the plot.
    out_dir : Path, optional
        Destination directory for saving the plot (default "phase_plots").
    fname : str, optional
        Basename (without extension) for the output PNG file (default "diagram").

    Returns
    -------
    None
        Saves the phase diagram to disk; does not return a value.
    """
    xrange, yrange = _axis_ranges(
        df, df, x_col=dex_col, y_col=peo_col, pad=2
    )

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = _figure_4x3(width_px=4200, height_px=1800, dpi=300)

    _, _, boundary_exp = plot_gmm_decision_regions(
        df,
        x_col=dex_col,
        y_col=peo_col,
        phase_col=true_phase_col,
        ax=ax,
        xrange=xrange,
        yrange=yrange,
        region_colors=["aquamarine", "lightsteelblue"],
        boundary_color="red",
        decision_alpha=1.0,
        plot_regions=True,
        decision_boundary_width=3,
    )

    plot_gmm_composition_phase(
        df,
        x_col=dex_col,
        y_col=peo_col,
        phase_col=pred_phase_col,
        ax=ax,
        point_cmap=["#FFFFCC", "dodgerblue"],
    )
    _, _, boundary_pred = plot_gmm_decision_regions(
        df,
        x_col=dex_col,
        y_col=peo_col,
        phase_col=pred_phase_col,
        ax=ax,
        xrange=xrange,
        yrange=yrange,
        plot_regions=False,
        boundary_color="blue",
        decision_boundary_width=2,
    )

    handles = [
        Patch(facecolor="lightsteelblue", edgecolor="black",
              label="Single-Phase (Experiment)"),
        Patch(facecolor="aquamarine", edgecolor="black",
              label="Two-Phase (Experiment)"),
        Line2D([], [], marker="^", linestyle="", color="black",
               markerfacecolor="#FFFFCC", markeredgecolor="black",
               markersize=10, label="Two-Phase (Model)"),
        Line2D([], [], marker="s", linestyle="", color="black",
               markerfacecolor="dodgerblue", markeredgecolor="black",
               markersize=10, label="Single-Phase (Model)"),
    ]
    if boundary_exp is not None:
        handles.append(Line2D([],[],color="red",lw=3,
                label="Decision Boundary"
            )
        )
    if boundary_pred is not None:
        handles.append(Line2D([],[],color="blue",lw=2,
                label="Decision Boundary"
            )
        )

    ax.legend(
        handles=handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        framealpha=0.8,
        fontsize=24,
    )

    _set_axis_style(ax, xrange, yrange)
    ax.set_xlabel(dex_col, fontsize=32, fontweight="bold", labelpad=20)
    ax.set_ylabel(peo_col, fontsize=32, fontweight="bold", labelpad=20)
    
    _apply_aspect(ax,equal_aspect=False)
    _format_ticks(ax,tick_fontsize=24,tick_weight="bold")
    
    png = Path(out_dir) / f"{fname}.png"
    fig.savefig(png, pad_inches=1.0, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Phase diagram saved -> {png}")