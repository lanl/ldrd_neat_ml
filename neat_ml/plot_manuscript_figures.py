from pathlib import Path

from neat_ml.utils import lib_plotting

if __name__ == "__main__":

    """
    A complete wrapper method to the plotting of all assocaited
    manuscript figures. The varibles defined below is used as
    function arguments for lib_plotting.plot_figures()
    """

    PHASE_COLS = ("Phase_Separation_1st", "Phase_Separation_2nd")
    FIG_3_CSV = Path("neat_ml/data/figure_data/Titration_Figures/")
    FIG_6_CSV = Path("neat_ml/data/figure_data/Binodal_Comparison_Figures/")
    CSV_PHASE_DIR = Path("neat_ml/data/Binary_Mixture_Phase_Information")
    OUT_DIR = Path("neat_ml/data/Figures_for_Manuscript")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MAT_MODEL_CSV = Path(
        "neat_ml/data/Binary_Mixture_Phase_Information/PEO8K_Sodium_Citrate_Composition.csv"
    )
    JSON_PATH = Path(
        "neat_ml/data/mathematical_model_parameters.json"
    )
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
