"""Geometric inversion: project 3-D points onto NURBS curves and surfaces."""

from __future__ import annotations

import copy

import numpy as np
from geomdl import NURBS, abstract, operations


class Curve2d_inv:
    """Find the parameter *t* ∈ [0, 1] on a 2-D NURBS curve closest to a query point."""

    def __init__(self, curve: NURBS.Curve) -> None:
        self.curve = curve
        self.loc3d = None
        self.loc2d = None

    @classmethod
    def from_parameters(cls, degree, control_points, knots) -> "Curve2d_inv":
        c = NURBS.Curve()
        c.degree = degree
        c.ctrlpts = control_points
        c.knotvector = knots
        return cls(c)

    @classmethod
    def from_Curve2d(cls, curve2d) -> "Curve2d_inv":
        return cls.from_parameters(curve2d.degree, curve2d.fix_ctp, curve2d.knotvector)

    # ------------------------------------------------------------------

    def is_valid_polygon(self, polygon: np.ndarray) -> bool:
        n = len(polygon) - 1
        for i in range(1, n):
            v01 = polygon[i + 1] - polygon[i - 1]
            vi0 = polygon[i] - polygon[i - 1]
            t1 = np.clip(np.dot(v01, vi0) / np.dot(v01, v01), 0, 1)
            V1 = polygon[i - 1] + t1 * v01
            V1Pi = polygon[i] - V1
            P_ = polygon[n] if i < n / 2 else polygon[0]
            v0p = P_ - polygon[i - 1]
            t2 = np.clip(np.dot(v01, v0p) / np.dot(v01, v01), 0, 1)
            V2 = polygon[i - 1] + t2 * v01
            V2P_ = P_ - V2
            if np.dot(V1Pi, V2P_) > 0:
                return False
        return True

    def is_flat_enough(self, polygon: np.ndarray, tolerance: float = 1e-3) -> bool:
        return all(
            np.linalg.norm(polygon[i] - polygon[0]) <= tolerance
            for i in range(1, len(polygon) - 1)
        )

    def is_nearest_point_on_endpoints(self, polygon: np.ndarray, test_point) -> bool:
        P0P = test_point - polygon[0]
        P0P1 = polygon[1] - polygon[0]
        PPn = polygon[-1] - test_point
        Pn_1Pn = polygon[-1] - polygon[-2]
        PnP0 = polygon[0] - polygon[-1]
        PnP = test_point - polygon[-1]
        R1 = np.dot(P0P, P0P1)
        R2 = np.dot(PPn, Pn_1Pn)
        R3 = np.dot(PnP0, PnP)
        R4 = np.dot(PnP0, P0P)
        return not (R1 > 0 and R2 > 0 or R3 * R4 < 0)

    def find_closest_point_on_bezier_curve(self, curve, test_point) -> float:
        pts = np.array(curve.evalpts)
        params = np.linspace(0, 1, curve.sample_size)
        return float(params[np.linalg.norm(pts - test_point, axis=1).argmin()])

    def subdivide_and_check(self, curve, test_point, candidates, param_range, tol):
        polygon = np.array(curve.ctrlpts)
        if self.is_valid_polygon(polygon):
            if self.is_nearest_point_on_endpoints(polygon, test_point):
                candidates += param_range
                return
            if self.is_flat_enough(polygon, tol):
                t = self.find_closest_point_on_bezier_curve(curve, test_point)
                candidates.append(param_range[0] + (param_range[1] - param_range[0]) * t)
                return
        left, right = operations.split_curve(curve, 0.5)
        mid = (param_range[0] + param_range[1]) / 2
        self.subdivide_and_check(left, test_point, candidates, [param_range[0], mid], tol)
        self.subdivide_and_check(right, test_point, candidates, [mid, param_range[1]], tol)

    def split_to_bezier(self, obj):
        curve = copy.deepcopy(obj)
        current_range = [0.0, 1.0]
        multi_range = [[0.0, 1.0]]
        multi_curve = []
        knots = curve.knotvector[curve.degree + 1: -(curve.degree + 1)]
        while knots:
            t = knots[0]
            left, curve = operations.split_curve(curve, param=t)
            multi_curve.append(left)
            abs_t = current_range[0] * (1 - t) + current_range[1] * t
            multi_range[-1][1] = abs_t
            multi_range.append([abs_t, 1.0])
            current_range = [abs_t, 1.0]
            knots = curve.knotvector[curve.degree + 1: -(curve.degree + 1)]
        multi_curve.append(curve)
        return multi_curve, multi_range

    def find_candidate_points(self, test_point, tolerance: float = 1e-3):
        bezier_curves, param_ranges = self.split_to_bezier(self.curve)
        candidates = []
        for bc, pr in zip(bezier_curves, param_ranges):
            self.subdivide_and_check(bc, test_point, candidates, pr, tolerance)
        return candidates

    def newton_raphson_refine(self, initial_guess, test_point, tol=1e-6, max_iter=100):
        u = initial_guess
        for _ in range(max_iter):
            derivs = self.curve.derivatives(u, order=2)
            pt, dpt, d2pt = derivs[0], derivs[1], derivs[2]
            r = np.array(pt) - np.array(test_point)
            f = np.dot(dpt, r)
            df = np.dot(dpt, dpt) + np.dot(r, d2pt)
            if abs(f) < tol:
                break
            u -= f / df
        return u

    def fit(self, test_point):
        self.loc3d = test_point
        candidates = self.find_candidate_points(test_point)
        best = min(candidates, key=lambda t: np.linalg.norm(
            np.array(self.curve.evaluate_single(t)) - np.array(test_point)
        ))
        self.loc2d = self.newton_raphson_refine(best, test_point)
        return self.loc2d

    def error(self) -> float:
        return float(np.linalg.norm(
            np.array(self.curve.evaluate_single(self.loc2d)) - np.array(self.loc3d)
        ))


# ---------------------------------------------------------------------------


class Surface3d_inv:
    """Project a 3-D point onto a NURBS surface to find its (u, v) parameters."""

    def __init__(self, surface: NURBS.Surface) -> None:
        self.surface = surface
        self.loc3d = None
        self.loc2d = None

    @classmethod
    def from_parameters(cls, u_degree, v_degree, ctpw, u_knots, v_knots, su, sv) -> "Surface3d_inv":
        s = NURBS.Surface()
        s.degree_u = u_degree
        s.degree_v = v_degree
        s.ctrlpts_size_u = su
        s.ctrlpts_size_v = sv
        s.ctrlptsw = ctpw
        s.knotvector_u = u_knots
        s.knotvector_v = v_knots
        return cls(s)

    @classmethod
    def from_Surface3d(cls, surface3d) -> "Surface3d_inv":
        ctp = surface3d.init_ctp
        w = surface3d.weight
        ctpw = np.concatenate([ctp * w[:, :, np.newaxis], w[:, :, np.newaxis]], -1).reshape(-1, 4).tolist()
        return cls.from_parameters(
            surface3d.degree_u, surface3d.degree_v, ctpw,
            surface3d.knotvector_u, surface3d.knotvector_v,
            surface3d.size_u, surface3d.size_v,
        )

    # ------------------------------------------------------------------

    def is_valid_polygon(self, polygon) -> bool:
        polygon = np.asarray(polygon)
        n = len(polygon) - 1
        for i in range(1, n):
            v01 = polygon[i + 1] - polygon[i - 1]
            vi0 = polygon[i] - polygon[i - 1]
            t1 = np.clip(np.dot(v01, vi0) / np.dot(v01, v01), 0, 1)
            V1 = polygon[i - 1] + t1 * v01
            V1Pi = polygon[i] - V1
            P_ = polygon[n] if i < n / 2 else polygon[0]
            v0p = P_ - polygon[i - 1]
            t2 = np.clip(np.dot(v01, v0p) / np.dot(v01, v01), 0, 1)
            V2 = polygon[i - 1] + t2 * v01
            V2P_ = P_ - V2
            if np.dot(V1Pi, V2P_) > 0:
                return False
        return True

    def is_valid_cp_net(self, control_points) -> bool:
        cp = np.asarray(control_points)
        n, m = cp.shape[:2]
        for j in range(m):
            if not self.is_valid_polygon(cp[:, j]):
                return False
        for i in range(n):
            if not self.is_valid_polygon(cp[i]):
                return False
        return True

    def is_flat_enough(self, polygon, tolerance: float = 1e-3) -> bool:
        polygon = np.asarray(polygon)
        return all(
            np.linalg.norm(polygon.reshape(-1, polygon.shape[-1])[i] - polygon.reshape(-1, polygon.shape[-1])[0]) <= tolerance
            for i in range(1, polygon.reshape(-1, polygon.shape[-1]).shape[0] - 1)
        )

    def find_closest_point_on_bezier_patch(self, surface, test_point):
        pts = np.array(surface.evalpts)
        su, sv = surface.sample_size
        params_u = np.linspace(0, 1, su)
        params_v = np.linspace(0, 1, sv)
        idx = int(np.linalg.norm(pts - test_point, axis=1).argmin())
        return params_u[idx // sv], params_v[idx % sv]

    def split_to_bezier(self, surf):
        surfs = operations.decompose_surface(surf)
        ku = np.unique(surf.knotvector_u)
        kv = np.unique(surf.knotvector_v)
        u_ranges = np.stack([ku[:-1], ku[1:]]).T
        v_ranges = np.stack([kv[:-1], kv[1:]]).T
        vi, ui = np.meshgrid(np.arange(len(kv) - 1), np.arange(len(ku) - 1))
        return surfs, u_ranges[ui].reshape(-1, 2), v_ranges[vi].reshape(-1, 2)

    def subdivide_and_check(self, subsurface, test_point, candidates, u_range, v_range, tol):
        cp = np.array(subsurface.ctrlpts).reshape(subsurface.ctrlpts_size_u, subsurface.ctrlpts_size_v, -1)[..., :3]
        if self.is_valid_cp_net(cp):
            if self.is_flat_enough(cp, tol):
                u, v = self.find_closest_point_on_bezier_patch(subsurface, test_point)
                candidates.append([
                    u_range[0] + (u_range[1] - u_range[0]) * u,
                    v_range[0] + (v_range[1] - v_range[0]) * v,
                ])
                return
        splits_u = operations.split_surface_u(subsurface, 0.5)
        sub4 = []
        for s in splits_u:
            sub4 += operations.split_surface_v(s, 0.5)
        u_mid = (u_range[0] + u_range[1]) / 2
        v_mid = (v_range[0] + v_range[1]) / 2
        sub_u_ranges = [[u_range[0], u_mid], [u_range[0], u_mid], [u_mid, u_range[1]], [u_mid, u_range[1]]]
        sub_v_ranges = [[v_range[0], v_mid], [v_mid, v_range[1]], [v_range[0], v_mid], [v_mid, v_range[1]]]
        for i, s in enumerate(sub4):
            self.subdivide_and_check(s, test_point, candidates, sub_u_ranges[i], sub_v_ranges[i], tol)

    def newton_raphson_refine(self, initial_guess, test_point, tol=1e-6, max_iter=100):
        u, v = initial_guess
        for _ in range(max_iter):
            skl = np.array(self.surface.derivatives(u, v, order=2))
            r = skl[0][0] - test_point
            du, dv = skl[1][0], skl[0][1]
            duu, duv, dvv = skl[2][0], skl[1][1], skl[0][2]
            kappa = [np.dot(du, r), np.dot(dv, r)]
            J = [
                [np.dot(du, du) + np.dot(r, duu), np.dot(du, dv) + np.dot(r, duv)],
                [np.dot(dv, du) + np.dot(r, duv), np.dot(dv, dv) + np.dot(r, dvv)],
            ]
            delta = np.linalg.solve(J, kappa)
            if np.all(np.abs(delta) < tol):
                break
            u = np.clip(u - delta[0], 0.0, 1.0)
            v = np.clip(v - delta[1], 0.0, 1.0)
        return u, v

    def fit(self, test_point):
        self.loc3d = test_point
        surfs, u_ranges, v_ranges = self.split_to_bezier(self.surface)
        candidates = []
        for s, ur, vr in zip(surfs, u_ranges, v_ranges):
            self.subdivide_and_check(s, test_point, candidates, ur, vr, 1e-3)
        best = min(candidates, key=lambda uv: np.linalg.norm(
            np.array(self.surface.evaluate_single(uv)) - np.array(test_point)
        ))
        self.loc2d = self.newton_raphson_refine(best, test_point)
        return self.loc2d

    def error(self) -> float:
        return float(np.linalg.norm(
            np.array(self.surface.evaluate_single(list(self.loc2d))) - np.array(self.loc3d)
        ))


# ---------------------------------------------------------------------------


def trim_edge_inversion(surface, trim, test_point):
    """Project *test_point* to *surface* then to *trim* edge.

    Parameters
    ----------
    surface:
        A :class:`~cadmap.geometry.nurbs.BsplineSurface`.
    trim:
        A :class:`~cadmap.geometry.nurbs.Curve2d` trim edge.
    test_point:
        3-D query point.

    Returns
    -------
    Parameter *t* on *trim* closest to *test_point*.
    """
    surf_fitter = Surface3d_inv.from_Surface3d(surface)
    edge_fitter = Curve2d_inv.from_Curve2d(trim)
    loc2d = surf_fitter.fit(test_point)
    loc_on_edge = edge_fitter.fit(loc2d)
    return loc_on_edge
