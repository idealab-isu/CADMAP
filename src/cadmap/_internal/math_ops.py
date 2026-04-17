"""Internal math helpers: winding numbers, barycentric, KNN, gradient utilities."""

from __future__ import annotations

from typing import Any, List

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# 2-D point-in-polygon (winding number)
# ---------------------------------------------------------------------------

def _is_left(p0, p1, p2):
    return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])


def wn_PnPoly(P, V) -> int:
    """Winding-number test: returns +1 (inside) or 0 (outside)."""
    wn = 0
    V = tuple(V[:]) + (V[0],)
    for i in range(len(V) - 1):
        if V[i][1] <= P[1]:
            if V[i + 1][1] > P[1]:
                if _is_left(V[i], V[i + 1], P) > 0:
                    wn += 1
        else:
            if V[i + 1][1] <= P[1]:
                if _is_left(V[i], V[i + 1], P) < 0:
                    wn -= 1
    return np.sign(wn)


def shoelace(x_y) -> float:
    """Signed polygon area (positive = CCW)."""
    x_y = np.array(x_y).reshape(-1, 2)
    x, y = x_y[:, 0], x_y[:, 1]
    return 0.5 * (np.sum(x * np.roll(y, -1)) - np.sum(y * np.roll(x, -1)))


def batch_wn(points: np.ndarray, face: Any) -> np.ndarray:
    """Return a boolean mask of shape ``points.shape[:-1]``.

    ``True`` where *points* (in parametric UV space) lie inside *all* trim
    wires of *face*.

    Parameters
    ----------
    points:
        Array of shape ``(…, 2)`` in [0, 1] × [0, 1] parametric space.
    face:
        A :class:`~cadmap.geometry.topology.Topology` node (face level).
    """
    from OCC.Core.gp import gp_Pnt2d
    from cadmap.geometry.nurbs import Curve2d  # avoid circular at module level

    wn_all = np.ones(points.shape[:-1], dtype=bool)

    for wire in face.descendent:
        curve_pts_all = []
        for edge in wire.descendent:
            c2d: Curve2d = edge.geom2d
            cpts = np.array(c2d.evaluate_list(np.arange(0, 1, 0.01)))
            curve_pts_all.append(cpts)

        poly = np.concatenate(curve_pts_all)
        wn = np.array([wn_PnPoly(p, poly) for p in points.reshape(-1, 2)])

        ccw = np.sign(shoelace(poly))
        wn = np.sign(wn)
        if ccw == -1:
            wn += 1  # CW wire → invert membership

        # Edge-on check: points exactly on the boundary are considered inside
        for idx, pt in enumerate(points.reshape(-1, 2)):
            pnt2d = gp_Pnt2d(float(pt[0]), float(pt[1]))
            for edge in wire.descendent:
                occ_curve = edge.geom2d._to_Geom2d_BSplineCurve()
                from OCC.Core.Geom2dAPI import Geom2dAPI_ProjectPointOnCurve
                proj = Geom2dAPI_ProjectPointOnCurve(pnt2d, occ_curve)
                if proj.NbPoints() > 0 and proj.LowerDistance() <= 1e-5:
                    wn[idx] = 1
                    break

        wn_all = np.logical_and(wn_all, wn.astype(bool).reshape(points.shape[:-1]))

    return wn_all


# ---------------------------------------------------------------------------
# Area / gradient utilities
# ---------------------------------------------------------------------------

def area_from_points(ptcloud) -> float:
    """Approximate surface area from a structured (N, M, 3) point grid."""
    if ptcloud.ndim != 3:
        raise TypeError("Input must be N×M×3.")
    if isinstance(ptcloud, torch.Tensor):
        lu = torch.sqrt(((ptcloud[:-1, :-1] - ptcloud[1:, :-1]) ** 2).sum(-1).squeeze())
        lv = torch.sqrt(((ptcloud[:-1, :-1] - ptcloud[:-1, 1:]) ** 2).sum(-1).squeeze())
        return torch.sum(lu * lv)
    lu = np.sqrt(((ptcloud[:-1, :-1] - ptcloud[1:, :-1]) ** 2).sum(-1).squeeze())
    lv = np.sqrt(((ptcloud[:-1, :-1] - ptcloud[:-1, 1:]) ** 2).sum(-1).squeeze())
    return float(np.sum(lu * lv))


def grid_avg_grad(input_grad: torch.Tensor, max_range: int, device: str) -> torch.Tensor:
    """Smooth a gradient field by local averaging over a ±*max_range* neighbourhood."""
    result = torch.zeros_like(input_grad)
    count = torch.zeros_like(input_grad)
    nu, nv, _ = input_grad.shape
    for di in range(-max_range, max_range + 1):
        for dj in range(-max_range, max_range + 1):
            i0, i1 = max(0, di), min(nu, nu + di)
            j0, j1 = max(0, dj), min(nv, nv + dj)
            si0, si1 = max(0, -di), min(nu, nu - di)
            sj0, sj1 = max(0, -dj), min(nv, nv - dj)
            result[i0:i1, j0:j1] += input_grad[si0:si1, sj0:sj1]
            count[i0:i1, j0:j1] += 1
    return result / count


def jitter_targets(targets: List[np.ndarray], scale: float = 0.001) -> List[np.ndarray]:
    """Add Gaussian noise to each point-cloud array."""
    return [pts + np.random.normal(0, scale, pts.shape) for pts in targets]


# ---------------------------------------------------------------------------
# K-nearest neighbours (torch-compatible)
# ---------------------------------------------------------------------------

class KNearestNeighbors:
    """Thin wrapper around sklearn ``NearestNeighbors`` that accepts torch tensors."""

    def __init__(self, n_neighbors: int = 5) -> None:
        self._nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
        self.n_neighbors = n_neighbors

    def fit(self, X: Any) -> "KNearestNeighbors":
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        self._nn.fit(X)
        return self

    def kneighbors(self, X: Any):
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = np.asarray(X)
        dists, indices = self._nn.kneighbors(X_np)
        return (
            torch.tensor(dists, dtype=torch.float32),
            torch.tensor(indices, dtype=torch.long),
        )
