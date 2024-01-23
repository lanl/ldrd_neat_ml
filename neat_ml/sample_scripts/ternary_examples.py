"""
Some sample plots of ternary phase diagrams
using our current plotting infrastructure.

The actual regression tests for plotting
are elsewhere, this is just for producing
visuals/slides, etc.
"""

from neat_ml import lib

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_excel("neat_ml/data/Hypothetical_ternary_phase_map.xlsx",
                   sheet_name=[0, 1],
                   header=1)
df_80_pt_wat = df[0].iloc[..., 4:-1]
le_80 = LabelEncoder()
y_80_pt_wat = le_80.fit_transform(df[0].iloc[..., -1])
le_20 = LabelEncoder()
y_20_pt_wat = le_20.fit_transform(df[1].iloc[..., -1])
df_20_pt_wat = df[1].iloc[..., 4:-1]

for df, y, plot_name in zip([df_80_pt_wat, df_20_pt_wat],
                            [y_80_pt_wat, y_20_pt_wat],
                            ["80_pt_wat_Mihee.png", "20_pt_wat_Mihee.png"]):
    actual_fig = lib.plot_tri_phase_diagram(X=df.to_numpy(),
                                            y=y,
                                            plot_path=".",
                                            plot_name=plot_name,
                                            bottom_label_z="Dextran (wt %)",
                                            right_label_y="PEO (wt %)",
                                            left_label_x="PEO-dextran block copolymer (wt%)",
                                            clockwise=False)

# for the Wikipedia Example:
# https://en.wikipedia.org/wiki/Ternary_plot#Example
X = np.array([[50, 20, 30],
              [10, 60, 30],
              [10, 30, 60]])
y = np.ones(3)
actual_fig = lib.plot_tri_phase_diagram(X=X,
                                        y=y,
                                        plot_path=".",
                                        plot_name="wikipedia_ex.png",
                                        bottom_label_z="Sand Separate (%)",
                                        right_label_y="Silt Separate (%)",
                                        left_label_x="Clay Separate (%)",
                                        clockwise=True)
