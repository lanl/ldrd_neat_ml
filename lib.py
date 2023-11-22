import numpy as np


# these features were considered potentially interesting
# for machine learning-based prediction of polymer
# properties by Mihee, Cesar, and Tyler during a discussion
# on Nov. 3/2023
features_of_interest = ["volume_fraction",
                        # temperature at which light transmission of
                        # polymer mixture changes (within biological range):
                        "temp_light_transmission", 
                        # polydispersity index (dispersity) -- different
                        # polymer lengths that result from imperfections in
                        # human synthesis/manufacturing processes:
                        "PDI",
                        # weight-average molar mass:
                        "M_w",
                        "backbone_or_torsion_angles",
                        "radius_of_gyration",
                        # these maps might be intra or even inter-molecular:
                        "contact_or_adjacency_maps",
                        # H-bond contacts in first solvation shell?
                        "h_bond_contacts",
                        # Mihee can measure interfacial tension
                        "interfacial_tension",
                        # Viscoscity may matter for cytometry, but not clear
                        # if it would matter for polymer phase separatation/
                        # microparticle formation:
                        "viscosity"]

def preprocess_data(df):
    # Accepts an input DataFrame and produces the typical
    # training (X) and prediction (y) values needed for ML
    X = df[["Dextran (wt%)",  "PEO (wt%)"]].to_numpy()
    y = df["Phase Separated"].to_numpy()
    y[y == "Yes"] = 1
    y[y == "No"] = 0
    y = y.astype(np.int32)
    return X, y


hyper_param_dict = {"rfc": {"max_depth": [1, 10, 100, None],
                            "min_samples_split": [2, 4, 10],
                            },
                    "xgb_class": {"n_estimators": [20, 100, 2_000],
                                  "subsample": [1, 0.9, 0.8],
                                  "colsample_bytree": [1, 0.9, 0.7]},
                    "xgb_dart": {"n_estimators": [20, 100, 300],
                                  "subsample": [1, 0.8],
                                  "colsample_bytree": [1, 0.9, 0.7]},
                    "svm": {"C": [1, 10],
                            "kernel": ["linear", "rbf"]},
                    }

def color_df(styler):
    # TODO: set vmin/vmax based on actual
    # range of values in DataFrame
    vmin = 0.8
    vmax = 1.0
    styler.background_gradient(axis=None, vmin=vmin, vmax=vmax, cmap="viridis")
    return styler
