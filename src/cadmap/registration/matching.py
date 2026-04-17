"""Surface-id matching and binary grid search for point-cloud-to-CAD registration."""

from __future__ import annotations

import warnings
from typing import List

import numpy as np
import torch
from pytorch3d.loss import chamfer_distance
from scipy.optimize import linear_sum_assignment


def match_surface_id(CAD: List[np.ndarray], ptcloud: List[np.ndarray]) -> List[List[int]]:
    """Match point-cloud patches to CAD surfaces by minimising Chamfer distance.

    Parameters
    ----------
    CAD:
        List of (N_i, 3) arrays, one per CAD surface.
    ptcloud:
        List of (M_j, 3) arrays, one per NDT point-cloud patch.

    Returns
    -------
    assignments:
        List of lists. ``assignments[i]`` contains the indices of ptcloud
        patches assigned to CAD surface *i*.
    """
    n_surf = len(CAD)
    n_ptc = len(ptcloud)
    dist = np.zeros((n_surf, n_ptc))

    for i, cad_pts in enumerate(CAD):
        for j, ptc_pts in enumerate(ptcloud):
            dist[i, j] = chamfer_distance(
                torch.tensor(cad_pts.reshape(-1, 3), dtype=torch.float).unsqueeze(0),
                torch.tensor(ptc_pts.reshape(-1, 3), dtype=torch.float).unsqueeze(0),
            )[0].item()

    # Hungarian assignment on a square cost matrix
    if n_surf < n_ptc:
        expanded = np.vstack([dist] * (n_ptc // n_surf + 1))[:n_ptc, :]
    else:
        expanded = dist

    row_ind, col_ind = linear_sum_assignment(expanded)
    assignments: List[List[int]] = [[] for _ in range(n_surf)]
    for r, c in zip(row_ind, col_ind):
        worker = r % n_surf
        if c < n_ptc:
            assignments[worker].append(c)

    return assignments


def binary_grid_search(
    face0_geomdl,
    loc_3d: np.ndarray,
    on_edge2d,
    grid_size: int = 10,
    depth: int = 0,
    prev_d=None,
):
    """Find the edge parameter *t* ∈ [0, 1] whose 3-D point is closest to *loc_3d*.

    Uses adaptive grid search with recursive refinement.

    Parameters
    ----------
    face0_geomdl:
        A geomdl NURBS surface.
    loc_3d:
        Target 3-D point.
    on_edge2d:
        A :class:`~cadmap.geometry.nurbs.Curve2d` or geomdl curve.
    grid_size:
        Number of samples per iteration.
    depth:
        Recursion depth (stops at 5).
    prev_d:
        Previous best squared distance (used to detect non-convergence).

    Returns
    -------
    [u, v] parametric coordinates, or ``None`` if search fails.
    """
    if depth >= 5:
        return None

    u = [0.0, 1.0]
    d = np.inf
    n_loop = 0
    d_delta = np.inf

    uv = None
    x_loc = 0

    while d >= 1e-8 and n_loop <= 200 and d_delta >= 1e-6:
        l_points = np.linspace(u[0], u[1], grid_size)
        uv_raw = np.array(on_edge2d.evaluate_list(l_points))
        uv_raw = np.clip(uv_raw, 0.0, 1.0)
        eval_pts = np.array(face0_geomdl.evaluate_list(uv_raw)).reshape(grid_size, 3)
        sq_dist = ((eval_pts - loc_3d) ** 2).sum(-1)
        d_new = sq_dist.min()
        x_loc = int(sq_dist.argmin())
        u = [
            l_points[max(x_loc - 1, 0)],
            l_points[min(x_loc + 1, grid_size - 1)],
        ]
        d_delta = abs(d - d_new)
        d = d_new
        n_loop += 1
        uv = uv_raw

    if prev_d is not None and abs(prev_d - d) < 1e-2:
        warnings.warn(f"Binary search not convergent. Final squared error: {d:.2E}")
        return uv[x_loc].tolist()

    if d >= 1e-5:
        ans = binary_grid_search(
            face0_geomdl, loc_3d, on_edge2d,
            grid_size=5 * grid_size, depth=depth + 1, prev_d=d,
        )
        if ans is None:
            warnings.warn(f"Binary search not convergent. Final squared error: {d:.2E}")
        else:
            return ans

    if d >= 1e-5:
        warnings.warn(f"Binary search not convergent. Final squared error: {d:.2E}")

    return uv[x_loc].tolist() if uv is not None else None
