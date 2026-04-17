"""Tests for cadmap._internal.math_ops — winding number, area helpers."""

import numpy as np
import pytest

from cadmap._internal.math_ops import wn_PnPoly, shoelace, area_from_points


class TestWindingNumber:
    def _square_polygon(self):
        return np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    def test_inside(self):
        poly = self._square_polygon()
        assert wn_PnPoly(np.array([0.5, 0.5]), poly) != 0

    def test_outside(self):
        poly = self._square_polygon()
        assert wn_PnPoly(np.array([2.0, 2.0]), poly) == 0


class TestShoelace:
    def test_unit_square_area(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        assert abs(shoelace(pts) - 1.0) < 1e-10

    def test_triangle_area(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        assert abs(shoelace(pts) - 0.5) < 1e-10


class TestAreaFromPoints:
    def test_planar_grid(self):
        # (2, 2, 3) structured grid of a 1x1 square in z=0
        pts = np.array([
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        ])
        area = area_from_points(pts)
        assert area > 0

    def test_rejects_2d_input(self):
        with pytest.raises(TypeError):
            area_from_points(np.zeros((4, 3)))
