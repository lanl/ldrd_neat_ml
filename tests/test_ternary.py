from neat_ml import lib

import pytest
import numpy as np
from numpy.testing import assert_allclose


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
    actual_fig = lib.plot_tri_phase_diagram(X,
                                            y,
                                            plot_path=tmp_path,
                                            bottom_label_z="Sand Separate (%)",
                                            right_label_y="Silt Separate (%)",
                                            left_label_x="Clay Separate (%)",
                                            clockwise=True)

    # first check that plot was produced
    assert expected_plot_file.is_file()
    axis = actual_fig.get_axes()[0]
    cs = axis.collections[0]
    actual_offsets = cs.get_offsets()
    expected_offsets = [[45.0, 43.30127018922193],
                        [65.0, 8.660254037844386],
                        [35.0, 8.660254037844386]]
    # this check appears to be sensitivie to the operations
    # inside `plot_tri_phase_diagram()`
    assert_allclose(actual_offsets, expected_offsets)


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
