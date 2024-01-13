from neat_ml import lib

import numpy as np


def test_ternary_phase_diagram(tmp_path):
    # based on the 3-sample example here:
    # https://en.wikipedia.org/wiki/Ternary_plot#Example
    # compare with the reference 3-point plot there,
    # and regression test it
    X = np.array([[50, 20, 30],
                  [10, 60, 30],
                  [10, 30, 60]])
    # I'll set sample 3 as "phase separated"
    # (doesn't really apply to the wiki case, but useful
    # for us)
    y = np.array([0, 0, 1])

    expected_plot_file = tmp_path / "ternary.png"
    lib.plot_tri_phase_diagram(X,
                               y,
                               plot_path=tmp_path,
                               bottom_label_z="Sand Separate (%)",
                               right_label_y="Silt Separate (%)",
                               left_label_x="Clay Separate (%)",
                               clockwise=True)

    # first check that plot was produced
    assert expected_plot_file.is_file()
