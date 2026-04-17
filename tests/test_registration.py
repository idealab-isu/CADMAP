"""Tests for cadmap.registration — surface matching and geometric inversion."""

import numpy as np
import pytest

from cadmap.registration.inversion import Curve2d_inv


class TestCurve2dInv:
    def _make_line_curve(self):
        from geomdl import NURBS
        c = NURBS.Curve()
        c.degree = 1
        c.ctrlpts = [[0.0, 0.0], [1.0, 1.0]]
        c.knotvector = [0.0, 0.0, 1.0, 1.0]
        return c

    def test_fit_midpoint(self):
        curve = self._make_line_curve()
        fitter = Curve2d_inv(curve)
        t = fitter.fit([0.5, 0.5])
        assert 0.0 <= t <= 1.0
        assert abs(t - 0.5) < 0.05

    def test_fit_endpoint(self):
        curve = self._make_line_curve()
        fitter = Curve2d_inv(curve)
        t = fitter.fit([0.0, 0.0])
        assert abs(t) < 0.1

    def test_error_is_small(self):
        curve = self._make_line_curve()
        fitter = Curve2d_inv(curve)
        fitter.fit([0.7, 0.7])
        assert fitter.error() < 1e-4
