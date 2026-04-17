"""Tests for cadmap.geometry — BsplineSurface, Curve2d, Topology."""

import numpy as np
import pytest

from cadmap.geometry.nurbs import BsplineSurface, Curve2d
from cadmap.geometry.topology import Topology


def _make_curve2d():
    return Curve2d(
        init_ctp=[[0.0, 0.0], [1.0, 1.0]],
        knotvector=[0.0, 0.0, 1.0, 1.0],
        curv_id="Curve2d_0",
        periodic=False,
    )


def _make_plane():
    return BsplineSurface(
        init_ctp=np.array([
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        ]),
        knotvector_u=[0.0, 0.0, 1.0, 1.0],
        knotvector_v=[0.0, 0.0, 1.0, 1.0],
        surface_id="Surface_0",
        ctp_start_index=0,
        periodic_u=False,
        periodic_v=False,
    )


class TestCurve2d:
    def test_evaluate_at_midpoint(self):
        c = _make_curve2d()
        # evaluate_list takes a list of t-parameters in [0, 1]
        pts = c.evaluate_list([0.0, 0.5, 1.0])
        assert len(pts) == 3
        assert len(pts[0]) == 2

    def test_evaluate2d_single(self):
        c = _make_curve2d()
        pt = c.evaluate2d_single(0.5)
        assert len(pt) == 2
        assert abs(pt[0] - 0.5) < 0.05

    def test_to_geomdl(self):
        c = _make_curve2d()
        g = c._to_geomdl()
        assert g.degree == 1
        assert len(g.evalpts) > 0


class TestBsplineSurface:
    def test_evaluate_list_shape(self):
        surf = _make_plane()
        pts = np.array(surf.evaluate_list(surf.init_ctp, [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]))
        assert pts.shape == (3, 3)

    def test_center_point(self):
        surf = _make_plane()
        pts = np.array(surf.evaluate_list(surf.init_ctp, [[0.5, 0.5]]))
        assert abs(pts[0, 0] - 0.5) < 1e-5
        assert abs(pts[0, 1] - 0.5) < 1e-5
        assert abs(pts[0, 2] - 0.0) < 1e-5


class TestTopology:
    def _make_node(self):
        class _MockShape:
            def IsSame(self, other):
                return self is other

        t = Topology.__new__(Topology)
        t.TopoDS = _MockShape()
        t.descendent = []
        t.ancestor = []
        t.geom = None
        t.geom2d = None
        return t

    def test_add_descendent(self):
        parent = self._make_node()
        child = self._make_node()
        parent.add_descendent(child)
        assert child in parent.descendent

    def test_add_ancestor(self):
        parent = self._make_node()
        child = self._make_node()
        child.add_ancestor(parent)
        assert parent in child.ancestor
