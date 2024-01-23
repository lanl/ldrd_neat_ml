from neat_ml import lib

import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose


@pytest.mark.parametrize("X, y, clockwise, expected", [
    # based on the 3-sample example here:
    # https://en.wikipedia.org/wiki/Ternary_plot#Example
    # compare with the reference 3-point plot there,
    # and regression test it
    # I'll set sample 3 as "phase separated"
    # (doesn't really apply to the wiki case, but useful
    # for us)
    (np.array([[50, 20, 30],
              [10, 60, 30],
              [10, 30, 60]]),
     np.array([0, 0, 1]),
     True,
     [[45.0, 43.30127018922193],
      [65.0, 8.660254037844386],
      [35.0, 8.660254037844386]],
    ),
    # Mihee synthetic data with 80 % water
    (pd.read_excel("neat_ml/data/Hypothetical_ternary_phase_map.xlsx",
                   sheet_name=[0],
                   header=1)[0].iloc[..., 4:-1].to_numpy(),
     np.ones(29),
     False,
    [[50.0,69.28203230275508],
     [55.0,60.6217782649107],
     [60.0,51.96152422706631],
     [65.0,43.30127018922193],
     [70.0,34.64101615137754],
     [75.0,25.980762113533157],
     [80.0,17.32050807568877],
     [45.0,60.6217782649107],
     [50.0,51.96152422706631],
     [55.0,43.30127018922193],
     [60.0,34.64101615137754],
     [65.0,25.980762113533157],
     [70.0,17.32050807568877],
     [75.0,8.660254037844386],
     [35.0,43.30127018922193],
     [40.0,34.64101615137754],
     [45.0,25.980762113533157],
     [50.0,17.32050807568877],
     [55.0,8.660254037844386],
     [60.0,0.0],
     [25.0,25.980762113533157],
     [30.0,17.32050807568877],
     [35.0,8.660254037844386],
     [40.0,0.0],
     [20.0,17.32050807568877],
     [25.0,8.660254037844386],
     [30.0,0.0],
     [15.0,8.660254037844386],
     [20.0,0.0]]
    ),
    # Mihee synthetic data with 20 % water
    (pd.read_excel("neat_ml/data/Hypothetical_ternary_phase_map.xlsx",
                   sheet_name=[1],
                   header=1)[1].iloc[..., 4:-1].to_numpy(),
     np.ones(43),
     False,
    [[50.0,69.28203230275508],
     [55.0,60.6217782649107],
     [60.0,51.96152422706631],
     [65.0,43.30127018922193],
     [70.0,34.64101615137754],
     [75.0,25.980762113533157],
     [80.0,17.32050807568877],
     [85.0,8.660254037844386],
     [45.0,60.6217782649107],
     [50.0,51.96152422706631],
     [55.0,43.30127018922193],
     [60.0,34.64101615137754],
     [65.0,25.980762113533157],
     [70.0,17.32050807568877],
     [75.0,8.660254037844386],
     [80.0,0.0],
     [40.0,51.96152422706631],
     [45.0,43.30127018922193],
     [50.0,34.64101615137754],
     [55.0,25.980762113533157],
     [60.0,17.32050807568877],
     [65.0,8.660254037844386],
     [70.0,0.0],
     [35.0,43.30127018922193],
     [40.0,34.64101615137754],
     [45.0,25.980762113533157],
     [50.0,17.32050807568877],
     [55.0,8.660254037844386],
     [60.0,0.0],
     [30.0,34.64101615137754],
     [35.0,25.980762113533157],
     [40.0,17.32050807568877],
     [45.0,8.660254037844386],
     [50.0,0.0],
     [25.0,25.980762113533157],
     [30.0,17.32050807568877],
     [35.0,8.660254037844386],
     [40.0,0.0],
     [20.0,17.32050807568877],
     [25.0,8.660254037844386],
     [30.0,0.0],
     [15.0,8.660254037844386],
     [20.0,0.0]],
    )
    ])
def test_ternary_phase_diagram(tmp_path, X, y, clockwise, expected):
    expected_plot_file = tmp_path / "ternary.png"
    actual_fig = lib.plot_tri_phase_diagram(X,
                                            y,
                                            plot_path=tmp_path,
                                            bottom_label_z="Sand Separate (%)",
                                            right_label_y="Silt Separate (%)",
                                            left_label_x="Clay Separate (%)",
                                            clockwise=clockwise)

    # first check that plot was produced
    assert expected_plot_file.is_file()
    axis = actual_fig.get_axes()[0]
    cs = axis.collections[0]
    actual_offsets = cs.get_offsets()
    # this check appears to be sensitivie to the operations
    # inside `plot_tri_phase_diagram()`
    assert_allclose(actual_offsets, expected)


@pytest.mark.parametrize("X, y, match", [
    # case that doesn't sum to constant value
    (np.array([[50, 20, 30],
               [10, 60, 30],
               [10, 30, 59]]),
     np.zeros(3),
     "do not sum"),
    # case with incorrect shape that sums to
    # a constant value
    (np.array([[50, 20, 30, 2],
               [10, 60, 30, 2],
               [10, 30, 60, 2]]),
     np.zeros(3),
     "three variables"),
    ])
def test_ternary_phase_diagram_bad_inputs(tmp_path, X, y, match):
    # according to: https://en.wikipedia.org/wiki/Ternary_plot
    # a ternary plot is a barycentric plot on three variables
    # which sum to a constant, so we should error out when
    # input data is non-conforming
    with pytest.raises(ValueError, match=match):
        lib.plot_tri_phase_diagram(X, y, plot_path=tmp_path)
