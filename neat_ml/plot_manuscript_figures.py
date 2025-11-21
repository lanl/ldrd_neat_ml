from pathlib import Path
from importlib.resources import files, as_file
from neat_ml.utils import lib_plotting

if __name__ == "__main__":

    """
    A complete wrapper for plotting all associated manuscript
    figures. The variables defined below are used as
    function arguments for lib_plotting.plot_figures()
    """
    with as_file(files("neat_ml.data")) as base_path:
        PHASE_COLS = ("Phase_Separation_1st", "Phase_Separation_2nd")
        FIG_3_CSV = base_path / "figure_data" / "Titration_Figures"
        FIG_6_CSV = base_path / "figure_data" / "Binodal_Comparison_Figures"
        CSV_PHASE_DIR = base_path / "Binary_Mixture_Phase_Information"
        OUT_DIR = Path("neat_ml/data/Figures_for_Manuscript")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        MAT_MODEL_CSV = base_path / "Binary_Mixture_Phase_Information" / "PEO8K_Sodium_Citrate_Composition.csv"
        JSON_PATH = base_path / "mathematical_model_parameters.json"
        MAT_MODEL_PNG = OUT_DIR / (
            "PEO8K_Sodium_Citrate_Phase_Diagram_Experiment_Literature_Comparison.png"
        )

    lib_plotting.plot_figures(
        titration_csv_dir=FIG_3_CSV,
        binodal_csv_dir=FIG_6_CSV,
        csv_phase_dir=CSV_PHASE_DIR,
        out_dir=OUT_DIR,
        mat_model_csv=MAT_MODEL_CSV,
        mat_model_png=MAT_MODEL_PNG,
        json_path=JSON_PATH,
        phase_cols=PHASE_COLS,
        xrange=[0, 21],
        yrange=[0, 38]
    )
