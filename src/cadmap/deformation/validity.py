"""Geometric admissibility diagnostics for deformed CAD and meshes.

Fitting accuracy and interface continuity say nothing about whether the deformed
surface is still a geometrically admissible solid.  A surface can match the
displacement field, meet its neighbours along every shared edge, and still fold
over or pass through itself.

**Two different validity tests.**  ``BRepCheck_Analyzer`` checks topology and
parametric consistency -- closed wires, SameParameter, pcurve ranges -- and does
**not** test for self-intersection.  ``BRepAlgoAPI_Check`` (the BOPAlgo argument
analyser, which is what FreeCAD's geometry check uses) does.  A shape can pass
the first and fail the second, so both are reported here and never conflated.

The central local measure is the **signed** area ratio

.. math::

    r_A(u,v) = \\frac{N_\\alpha(u,v)\\cdot N_0(u,v)}{\\lVert N_0(u,v)\\rVert^2},
    \\qquad N = S_u \\times S_v ,

which is positive where orientation is preserved, near zero where the surface
collapses, and **negative where it folds**.  The unsigned magnitude
:math:`\\lVert N_\\alpha\\rVert` cannot detect inversion and must not be used for
this purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "MeshValidity",
    "mesh_validity",
    "triangle_self_intersections",
    "BRepStatus",
    "brep_status",
    "FaceFoldStats",
    "signed_area_ratio",
    "homotopy_shape",
]


# ---------------------------------------------------------------------------
# Target-mesh validity (plan section 3)
# ---------------------------------------------------------------------------

@dataclass
class MeshValidity:
    """Element quality of a deformed mesh relative to its undeformed state."""

    n_triangles: int
    min_area_original: float
    min_area_deformed: float
    min_area_ratio: float
    n_near_degenerate: int
    min_normal_cos: float
    n_flipped: int
    n_self_intersections: int
    self_intersection_pairs: List[Tuple[int, int]] = field(default_factory=list)


def _triangle_normals(nodes: np.ndarray, tris: np.ndarray) -> np.ndarray:
    p = nodes[tris]
    return np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])


def mesh_validity(
    nodes_original: np.ndarray,
    nodes_deformed: np.ndarray,
    triangles: np.ndarray,
    degenerate_ratio: float = 1e-6,
    check_self_intersection: bool = True,
) -> MeshValidity:
    """Element inversion, collapse and self-intersection of the deformed mesh.

    If the prescribed field already inverts or self-intersects the mesh, no
    valid CAD can reproduce it exactly, and the problem changes from "fit the
    field" to "find the closest admissible deformation".
    """
    X0 = np.asarray(nodes_original, dtype=float)
    X1 = np.asarray(nodes_deformed, dtype=float)
    T = np.asarray(triangles, dtype=np.int64)

    n0 = _triangle_normals(X0, T)
    n1 = _triangle_normals(X1, T)
    a0 = 0.5 * np.linalg.norm(n0, axis=1)
    a1 = 0.5 * np.linalg.norm(n1, axis=1)

    safe = np.maximum(a0, 1e-300)
    ratio = a1 / safe
    denom = np.maximum(np.linalg.norm(n0, axis=1) * np.linalg.norm(n1, axis=1), 1e-300)
    cos = np.einsum("ij,ij->i", n0, n1) / denom

    pairs: List[Tuple[int, int]] = []
    if check_self_intersection:
        pairs = triangle_self_intersections(X1, T)

    return MeshValidity(
        n_triangles=len(T),
        min_area_original=float(a0.min()),
        min_area_deformed=float(a1.min()),
        min_area_ratio=float(ratio.min()),
        n_near_degenerate=int((ratio < degenerate_ratio).sum()),
        min_normal_cos=float(cos.min()),
        n_flipped=int((cos <= 0.0).sum()),
        n_self_intersections=len(pairs),
        self_intersection_pairs=pairs[:50],
    )


def _tri_tri_intersect(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Separating-axis test for batches of triangle pairs.

    ``A`` and ``B`` are ``(n, 3, 3)``.  Returns a boolean array: True where no
    separating axis exists, i.e. the triangles intersect or touch.
    """
    axes = []
    nA = np.cross(A[:, 1] - A[:, 0], A[:, 2] - A[:, 0])
    nB = np.cross(B[:, 1] - B[:, 0], B[:, 2] - B[:, 0])
    axes.append(nA)
    axes.append(nB)
    for i in range(3):
        ea = A[:, (i + 1) % 3] - A[:, i]
        for j in range(3):
            eb = B[:, (j + 1) % 3] - B[:, j]
            axes.append(np.cross(ea, eb))

    separated = np.zeros(len(A), dtype=bool)
    for axis in axes:
        norm = np.linalg.norm(axis, axis=1)
        ok = norm > 1e-12                      # skip degenerate axes
        if not ok.any():
            continue
        u = np.where(ok[:, None], axis / np.maximum(norm, 1e-300)[:, None], 0.0)
        pa = np.einsum("nij,nj->ni", A, u)
        pb = np.einsum("nij,nj->ni", B, u)
        gap = (pa.min(axis=1) > pb.max(axis=1) + 1e-12) | \
              (pb.min(axis=1) > pa.max(axis=1) + 1e-12)
        separated |= (gap & ok)
    return ~separated


def triangle_self_intersections(
    nodes: np.ndarray,
    triangles: np.ndarray,
    max_pairs: int = 2_000_000,
) -> List[Tuple[int, int]]:
    """Non-adjacent intersecting triangle pairs of one triangulation.

    Triangles sharing a vertex are excluded: they meet by construction.
    """
    from scipy.spatial import cKDTree

    X = np.asarray(nodes, dtype=float)
    T = np.asarray(triangles, dtype=np.int64)
    tri = X[T]
    centroid = tri.mean(axis=1)
    radius = np.linalg.norm(tri - centroid[:, None, :], axis=2).max(axis=1)

    # Query per triangle with its OWN radius, in batches.  Using a single global
    # radius (2 * radius.max()) would make one oversized triangle turn this into
    # an O(n^2) allocation -- billions of pairs -- which exhausts memory before
    # any guard can fire.
    tree = cKDTree(centroid)
    keep: List[np.ndarray] = []
    total = 0
    batch = 2048
    for lo in range(0, len(tri), batch):
        hi = min(lo + batch, len(tri))
        idx = np.arange(lo, hi)
        neigh = tree.query_ball_point(centroid[idx], r=radius[idx] + radius.max())
        for local, others in zip(idx, neigh):
            others = np.asarray(others, dtype=np.int64)
            others = others[others > local]
            if not len(others):
                continue
            close = (np.linalg.norm(centroid[others] - centroid[local], axis=1)
                     <= radius[others] + radius[local])
            others = others[close]
            if not len(others):
                continue
            total += len(others)
            if total > max_pairs:
                raise MemoryError(
                    f"more than {max_pairs} candidate triangle pairs; "
                    f"the tessellation is too dense or has oversized triangles"
                )
            keep.append(np.stack([np.full(len(others), local), others], axis=1))
    if not keep:
        return []
    candidates = np.concatenate(keep)

    i, j = candidates[:, 0], candidates[:, 1]
    shares_vertex = np.zeros(len(candidates), dtype=bool)
    for a in range(3):
        for b in range(3):
            shares_vertex |= (T[i, a] == T[j, b])
    candidates = candidates[~shares_vertex]
    if not len(candidates):
        return []

    i, j = candidates[:, 0], candidates[:, 1]
    hit = _tri_tri_intersect(tri[i], tri[j])
    return [(int(a), int(b)) for a, b in candidates[hit]]


# ---------------------------------------------------------------------------
# B-Rep status (plan sections 4, 6)
# ---------------------------------------------------------------------------

@dataclass
class BRepStatus:
    """Both validity tests, kept separate because they check different things."""

    topologically_valid: bool          # BRepCheck_Analyzer
    boolean_valid: bool                # BRepAlgoAPI_Check (includes self-intersection)
    n_shells: int
    n_closed_shells: int
    n_invalid_faces: int
    n_invalid_edges: int
    n_faulty_results: int
    invalid_faces: List[int] = field(default_factory=list)


def brep_status(shape, faces: Optional[Sequence] = None) -> BRepStatus:
    """Validity of a B-Rep under both OpenCascade checks."""
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Check
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods

    from cadmap.conversion.export import is_closed_shell

    analyzer = BRepCheck_Analyzer(shape)

    shells, explorer = [], TopExp_Explorer(shape, TopAbs_SHELL)
    while explorer.More():
        shells.append(topods.Shell(explorer.Current()))
        explorer.Next()

    invalid_faces: List[int] = []
    idx, explorer = 0, TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        if not analyzer.IsValid(topods.Face(explorer.Current())):
            invalid_faces.append(idx)
        idx += 1
        explorer.Next()

    n_invalid_edges = 0
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while explorer.More():
        if not analyzer.IsValid(topods.Edge(explorer.Current())):
            n_invalid_edges += 1
        explorer.Next()

    # The shape-taking constructor already performs the complete check.
    check = BRepAlgoAPI_Check(shape)
    result = check.Result()
    try:
        n_faulty = int(result.Size())
    except Exception:
        n_faulty = len(list(result))

    return BRepStatus(
        topologically_valid=bool(analyzer.IsValid()),
        boolean_valid=bool(check.IsValid()),
        n_shells=len(shells),
        n_closed_shells=sum(is_closed_shell(s) for s in shells),
        n_invalid_faces=len(invalid_faces),
        n_invalid_edges=n_invalid_edges,
        n_faulty_results=n_faulty,
        invalid_faces=invalid_faces,
    )


# ---------------------------------------------------------------------------
# Local fold diagnostic (plan sections 7, 8)
# ---------------------------------------------------------------------------

@dataclass
class FaceFoldStats:
    """Signed area ratio statistics over one face's trimmed region."""

    face: int
    n_samples: int
    min_r: float
    p01_r: float
    p05_r: float
    n_nonpositive: int
    worst_uv: Optional[Tuple[float, float]]
    worst_point: Optional[np.ndarray]


def _surface_normals(surf, ctp: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """``S_u x S_v`` at each ``uv`` for the given control net."""
    from OCC.Core.gp import gp_Pnt, gp_Vec

    geom = surf._to_geomsurface(ctp)
    out = np.empty((len(uv), 3))
    p, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
    for k, (u, v) in enumerate(uv):
        geom.D1(float(u), float(v), p, du, dv)
        n = du.Crossed(dv)
        out[k] = (n.X(), n.Y(), n.Z())
    return out


def signed_area_ratio(
    solid,
    control_point_delta: np.ndarray,
    alpha: float = 1.0,
    per_ctp: float = 3.0,
    min_side: int = 8,
    max_side: int = 40,
) -> List[FaceFoldStats]:
    """Per-face signed area ratio over the trimmed region.

    ``r_A > 0`` orientation preserved, ``r_A -> 0`` local collapse,
    ``r_A < 0`` fold.  Using only ``||N_alpha||`` would miss inversion entirely.
    """
    from cadmap.deformation.mesh_transfer import _grid_inside_trim, _trim_classifier

    dP = np.asarray(control_point_delta, dtype=float)
    out: List[FaceFoldStats] = []

    for face_idx, face in enumerate(solid.descendent):
        surf = face.geom
        uv = _grid_inside_trim(surf, _trim_classifier(face), per_ctp, min_side, max_side)
        if not len(uv):
            continue
        start, stop = surf.ctp_index
        ctp0 = surf.init_ctp.reshape(-1, 3)
        ctpa = ctp0 + alpha * dP[start:stop]

        N0 = _surface_normals(surf, ctp0, uv)
        Na = _surface_normals(surf, ctpa, uv)
        denom = np.einsum("ij,ij->i", N0, N0)
        good = denom > 1e-300
        r = np.full(len(uv), np.nan)
        r[good] = np.einsum("ij,ij->i", Na[good], N0[good]) / denom[good]
        r = r[np.isfinite(r)]
        uv_ok = uv[np.isfinite(np.where(good, 1.0, np.nan))] if good.any() else uv
        if not len(r):
            continue

        worst = int(np.argmin(r))
        out.append(FaceFoldStats(
            face=face_idx,
            n_samples=len(r),
            min_r=float(r.min()),
            p01_r=float(np.percentile(r, 1)),
            p05_r=float(np.percentile(r, 5)),
            n_nonpositive=int((r <= 0.0).sum()),
            worst_uv=(float(uv_ok[worst][0]), float(uv_ok[worst][1]))
            if worst < len(uv_ok) else None,
            worst_point=None,
        ))
    return out


# ---------------------------------------------------------------------------
# Homotopy (plan sections 4, 5)
# ---------------------------------------------------------------------------

def homotopy_shape(solid, control_point_delta: np.ndarray, alpha: float,
                   sew_tol: float = 1e-6):
    """Rebuild the B-Rep at ``P(alpha) = P0 + alpha * dP``."""
    from cadmap.conversion.export import solid_to_occ_shape

    dP = np.asarray(control_point_delta, dtype=float)
    ctp0 = np.concatenate([f.geom.init_ctp.reshape(-1, 3) for f in solid.descendent])
    return solid_to_occ_shape(solid, ctp0 + alpha * dP, tol=sew_tol)
