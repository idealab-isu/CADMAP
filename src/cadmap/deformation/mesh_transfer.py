"""Mesh-guided NURBS deformation: drive a CAD solid with a mesh displacement field.

Given an as-designed surface mesh whose nodes lie on the CAD faces, and the same
mesh in a deformed state (identical node ordering), this module deforms the NURBS
CAD model so it follows the mesh's displacement field.  Nothing is fitted to a
sampled point cloud and no closest-point correspondence is searched for: the
mesh already tells us which point went where.

The method rests on one fact -- a NURBS surface is *linear* in its control
points::

    S(u, v; P) = sum_ij  R_ij(u, v) P_ij          R_ij = N_i(u) N_j(v) w_ij / W(u, v)

so transporting the displacement field onto each face's parameter domain turns
the whole deformation into a single linear least-squares problem, with the C0
continuity conditions at shared edges as *linear equality constraints*.  Solving
in a basis of the constraint null space satisfies continuity exactly rather than
approximately.

Pipeline::

    align   = rigid_transform(mesh_nodes, fit_frame_nodes)   # frames differ
    prints  = invert_nodes(solid, nodes)                     # node -> (face, u, v)
    system  = build_data_system(solid, prints, displacement)  # A dP = d
    rows    = build_continuity_rows(solid)                   # C dP = 0
    dP      = solve_constrained(system, rows)                # one solve, 3 RHS
    shape   = deformed_occ_shape(solid, dP)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from geomdl import helpers

from cadmap.geometry.topology import Topology

__all__ = [
    "rigid_transform",
    "RigidTransform",
    "surface_basis_matrix",
    "NodeFootprint",
    "invert_nodes",
    "build_data_system",
    "build_smoothing_rows",
    "build_continuity_rows",
    "solve_constrained",
    "control_point_count",
    "deform_solid",
    "DeformationResult",
]


# ---------------------------------------------------------------------------
# Rigid alignment between the mesh archive frames
# ---------------------------------------------------------------------------

@dataclass
class RigidTransform:
    """``x -> R @ x + t``, with the residual of the fit that produced it."""

    R: np.ndarray
    t: np.ndarray
    residual: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        return (self.R @ np.asarray(points, dtype=float).T).T + self.t

    def apply_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Rotate direction vectors (displacements), ignoring the translation."""
        return (self.R @ np.asarray(vectors, dtype=float).T).T

    def inverse(self) -> "RigidTransform":
        return RigidTransform(self.R.T, -self.R.T @ self.t, self.residual)

    @property
    def rotation_angle_deg(self) -> float:
        return float(np.degrees(np.arccos(np.clip((np.trace(self.R) - 1.0) / 2.0, -1.0, 1.0))))


def rigid_transform(source: np.ndarray, target: np.ndarray) -> RigidTransform:
    """Best rigid transform mapping *source* onto *target* (Kabsch, paired rows).

    Used to reconcile the two archives: the modal mesh lives in the CAD frame
    while the fitted mesh lives in the measurement frame, related by a pure
    rotation and translation.
    """
    A = np.asarray(source, dtype=float)
    B = np.asarray(target, dtype=float)
    if A.shape != B.shape:
        raise ValueError(f"paired point sets must match in shape: {A.shape} vs {B.shape}")

    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cb - R @ ca
    residual = np.linalg.norm((R @ A.T).T + t - B, axis=1)
    return RigidTransform(R, t, residual)


# ---------------------------------------------------------------------------
# Rational basis functions
# ---------------------------------------------------------------------------

def _basis_row(degree: int, knotvector: np.ndarray, n_ctp: int, t: float) -> Tuple[int, List[float]]:
    """Return ``(span, basis_values)`` for one parameter, clamped to the valid range."""
    kv = np.asarray(knotvector, dtype=float).tolist()
    # An unclamped knot vector (produced when a periodic surface is unwrapped)
    # is only valid on [kv[degree], kv[n_ctp]].
    lo, hi = kv[degree], kv[n_ctp]
    t = float(min(max(t, lo), hi))
    span = helpers.find_span_linear(degree, kv, n_ctp, t)
    return span, helpers.basis_function(degree, kv, span, t)


def surface_basis_matrix(surf, uv: np.ndarray) -> np.ndarray:
    """Rational basis matrix ``A`` with ``A @ P.reshape(-1, 3) == S(uv)``.

    Rows follow *uv*; columns follow the control net in ``init_ctp`` order
    (``size_u`` major, ``size_v`` minor), so a column block lines up with the
    face's slice of the global control-point array.
    """
    uv = np.atleast_2d(np.asarray(uv, dtype=float))
    n_u, n_v = surf.size_u, surf.size_v
    weight = np.asarray(surf.weight, dtype=float).reshape(n_u, n_v)

    A = np.zeros((len(uv), n_u * n_v), dtype=float)
    for row, (u, v) in enumerate(uv):
        span_u, bu = _basis_row(surf.degree_u, surf.knotvector_u, n_u, u)
        span_v, bv = _basis_row(surf.degree_v, surf.knotvector_v, n_v, v)
        i0 = span_u - surf.degree_u
        j0 = span_v - surf.degree_v

        idx_u = np.arange(i0, i0 + surf.degree_u + 1)
        idx_v = np.arange(j0, j0 + surf.degree_v + 1)
        # Outer product of the two univariate bases, weighted and normalised.
        block = np.outer(np.asarray(bu), np.asarray(bv)) * weight[np.ix_(idx_u, idx_v)]
        total = block.sum()
        if total == 0.0:
            continue
        block /= total
        cols = (idx_u[:, None] * n_v + idx_v[None, :]).ravel()
        A[row, cols] = block.ravel()
    return A


# ---------------------------------------------------------------------------
# Node inversion: mesh node -> (face, u, v)
# ---------------------------------------------------------------------------

@dataclass
class NodeFootprint:
    """A mesh node located on a CAD face at parameters ``(u, v)``."""

    node: int
    face: int
    u: float
    v: float
    distance: float


def invert_nodes(
    solid: Topology,
    nodes: np.ndarray,
    tol: float = 1e-6,
    bbox_margin: float = 1e-3,
    occ_faces: Optional[Sequence] = None,
) -> List[NodeFootprint]:
    """Locate every mesh node on the CAD faces it lies on.

    A node on a shared edge lands on two or more faces and yields one footprint
    per face, which is what makes the transported field agree across that edge.
    Parameters are returned in the ``[0, 1]`` convention used by
    :class:`~cadmap.geometry.nurbs.BsplineSurface`.

    Nodes further than *tol* from every face are skipped; the caller should
    check the returned count against ``len(nodes)``.
    """
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.BRepTopAdaptor import BRepTopAdaptor_FClass2d
    from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCC.Core.gp import gp_Pnt, gp_Pnt2d
    from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON

    from cadmap.conversion.export import build_occ_face

    nodes = np.asarray(nodes, dtype=float)
    if occ_faces is None:
        occ_faces = [build_occ_face(face) for face in solid.descendent]

    surfaces, classifiers, boxes = [], [], []
    for occ_face, face_topo in zip(occ_faces, solid.descendent):
        surfaces.append(face_topo.geom._to_geomsurface())
        classifiers.append(BRepTopAdaptor_FClass2d(occ_face, tol))
        box = Bnd_Box()
        brepbndlib.Add(occ_face, box)
        box.Enlarge(max(tol, bbox_margin))
        boxes.append(box)

    footprints: List[NodeFootprint] = []
    for n, p in enumerate(nodes):
        pnt = gp_Pnt(float(p[0]), float(p[1]), float(p[2]))
        for f, (surface, classifier, box) in enumerate(zip(surfaces, classifiers, boxes)):
            if box.IsOut(pnt):
                continue
            proj = GeomAPI_ProjectPointOnSurf(pnt, surface)
            if not proj.NbPoints() or proj.LowerDistance() > tol:
                continue
            u, v = proj.LowerDistanceParameters()
            if classifier.Perform(gp_Pnt2d(u, v)) not in (TopAbs_IN, TopAbs_ON):
                continue
            footprints.append(NodeFootprint(n, f, float(u), float(v), float(proj.LowerDistance())))
    return footprints


# ---------------------------------------------------------------------------
# Linear systems
# ---------------------------------------------------------------------------

def control_point_count(solid: Topology) -> int:
    """Total number of control points across all faces."""
    return sum(f.geom.size_u * f.geom.size_v for f in solid.descendent)


def build_data_system(
    solid: Topology,
    footprints: Sequence[NodeFootprint],
    displacement: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble ``A`` and ``d`` so that ``A @ dP ~= d`` reproduces the field.

    One row per footprint; ``d`` holds the nodal displacement of that node.
    Because the basis is scalar, the same ``A`` serves all three coordinates.
    """
    displacement = np.asarray(displacement, dtype=float)
    n_cols = control_point_count(solid)

    by_face: Dict[int, List[NodeFootprint]] = {}
    for fp in footprints:
        by_face.setdefault(fp.face, []).append(fp)

    A = np.zeros((len(footprints), n_cols), dtype=float)
    d = np.zeros((len(footprints), 3), dtype=float)

    row = 0
    for face_idx, prints in by_face.items():
        surf = solid.descendent[face_idx].geom
        start, stop = surf.ctp_index
        uv = np.array([[fp.u, fp.v] for fp in prints])
        block = surface_basis_matrix(surf, uv)
        rows = slice(row, row + len(prints))
        A[rows, start:stop] = block
        d[rows] = displacement[[fp.node for fp in prints]]
        row += len(prints)
    return A, d


def build_smoothing_rows(solid: Topology, weight: float = 1.0) -> np.ndarray:
    """Second-difference rows on each control net, for underdetermined faces.

    Faces carrying only a handful of mesh nodes (some have as few as one) cannot
    pin down their control net on data alone; penalising curvature of the
    *displacement* net keeps those faces smooth without biasing them away from
    the data where data exists.
    """
    n_cols = control_point_count(solid)
    rows: List[np.ndarray] = []

    for face in solid.descendent:
        surf = face.geom
        start, _ = surf.ctp_index
        n_u, n_v = surf.size_u, surf.size_v

        def col(i: int, j: int) -> int:
            return start + i * n_v + j

        for i in range(n_u):
            for j in range(1, n_v - 1):
                r = np.zeros(n_cols)
                r[col(i, j - 1)] = weight
                r[col(i, j)] = -2.0 * weight
                r[col(i, j + 1)] = weight
                rows.append(r)
        for j in range(n_v):
            for i in range(1, n_u - 1):
                r = np.zeros(n_cols)
                r[col(i - 1, j)] = weight
                r[col(i, j)] = -2.0 * weight
                r[col(i + 1, j)] = weight
                rows.append(r)

    if not rows:
        return np.zeros((0, n_cols))
    return np.array(rows)


@dataclass
class InterfacePair:
    """Matched samples along one shared edge, as seen from both faces.

    ``uv_a`` and ``uv_b`` address the *same physical points* on the two faces:
    the edge is sampled on face ``a`` and each 3-D sample is inverted onto face
    ``b``.  Pairing by equal parameter fractions instead would match different
    points, because the two pcurves of one edge carry independent
    parameterisations.
    """

    curve_id: str
    face_a: int
    face_b: int
    uv_a: np.ndarray          # (n, 2)
    uv_b: np.ndarray          # (n, 2)
    points: np.ndarray        # (n, 3) undeformed positions


def interface_pairs(
    solid: Topology,
    sample_size: int = 24,
    project_tol: float = 1e-7,
) -> List[InterfacePair]:
    """Matched point pairs along every shared edge.

    Shared by the C0 constraint assembly and by the interface-gap metric, so
    both speak about exactly the same geometry.  The metric must sample more
    densely than the constraint: measuring the gap only where continuity was
    collocated is circular, since it is ~0 there by construction.
    """
    from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.BRep import BRep_Tool

    shared: Dict[str, List[Tuple[int, object]]] = {}
    edge_tolerances: Dict[str, float] = {}
    for face_idx, face in enumerate(solid.descendent):
        for wire in face.descendent:
            for edge in wire.descendent:
                if edge.geom is None or edge.geom2d is None:
                    continue
                shared.setdefault(edge.geom.curv_id, []).append((face_idx, edge.geom2d))
                edge_tolerances[edge.geom.curv_id] = max(
                    edge_tolerances.get(edge.geom.curv_id, project_tol),
                    # Each face's pcurve may deviate from the shared 3-D
                    # edge by its tolerance; their mutual distance can be
                    # twice that tolerance (triangle inequality).
                    2.0 * float(BRep_Tool.Tolerance(edge.TopoDS)),
                )

    geom_cache: Dict[int, object] = {}

    def occ_surface(face_idx: int):
        if face_idx not in geom_cache:
            geom_cache[face_idx] = solid.descendent[face_idx].geom._to_geomsurface()
        return geom_cache[face_idx]

    ts = np.linspace(0.0, 1.0, sample_size)
    pairs: List[InterfacePair] = []

    for curve_id, refs in shared.items():
        if len(refs) < 2:
            continue

        base_idx, base_curve = refs[0]
        base_surf = solid.descendent[base_idx].geom
        base_uv = np.clip(np.array(base_curve.evaluate_list(ts.tolist()), dtype=float), 0.0, 1.0)
        base_pts = np.asarray(
            base_surf.evaluate_list(base_surf.init_ctp, base_uv.tolist()), dtype=float
        )

        for other_idx, _ in refs[1:]:
            surface = occ_surface(other_idx)
            other_uv = np.empty_like(base_uv)
            keep = np.zeros(len(base_uv), dtype=bool)
            for k, p in enumerate(base_pts):
                proj = GeomAPI_ProjectPointOnSurf(gp_Pnt(*(float(c) for c in p)), surface)
                # STEP pcurves on adjacent faces agree within the original
                # edge's tolerance, which may exceed project_tol. Dropping
                # those samples silently removes necessary C0 constraints.
                if not proj.NbPoints() or proj.LowerDistance() > edge_tolerances[curve_id]:
                    raise ValueError(
                        f"Shared edge {curve_id} cannot be paired within its "
                        f"source tolerance {edge_tolerances[curve_id]:g}"
                    )
                u, v = proj.LowerDistanceParameters()
                other_uv[k] = (u, v)
                keep[k] = True
            if not keep.any():
                continue
            pairs.append(InterfacePair(
                curve_id=curve_id, face_a=base_idx, face_b=other_idx,
                uv_a=base_uv[keep], uv_b=other_uv[keep], points=base_pts[keep],
            ))
    return pairs


def build_continuity_rows(
    solid: Topology,
    sample_size: int = 24,
    project_tol: float = 1e-7,
    pairs: Optional[Sequence[InterfacePair]] = None,
) -> np.ndarray:
    """Rows ``C`` enforcing C0 continuity along every shared edge.

    For each matched pair along a shared edge the two faces must evaluate to the
    same point, which in control-point space is the linear condition
    ``R_A dP_A - R_B dP_B = 0``.  The same scalar row applies to each coordinate.
    """
    n_cols = control_point_count(solid)
    if pairs is None:
        pairs = interface_pairs(solid, sample_size=sample_size, project_tol=project_tol)

    rows: List[np.ndarray] = []
    for pair in pairs:
        surf_a = solid.descendent[pair.face_a].geom
        surf_b = solid.descendent[pair.face_b].geom
        A_a = surface_basis_matrix(surf_a, pair.uv_a)
        A_b = surface_basis_matrix(surf_b, pair.uv_b)
        a0, a1 = surf_a.ctp_index
        b0, b1 = surf_b.ctp_index
        for k in range(len(pair.uv_a)):
            r = np.zeros(n_cols)
            r[a0:a1] += A_a[k]
            r[b0:b1] -= A_b[k]
            rows.append(r)

    if not rows:
        return np.zeros((0, n_cols))
    return np.array(rows)


def interface_gaps(
    solid: Topology,
    control_point_delta: np.ndarray,
    pairs: Optional[Sequence[InterfacePair]] = None,
    sample_size: int = 200,
) -> np.ndarray:
    """Distance between the two deformed faces along every shared edge.

    ``g = || v_a(s) - v_b(s) ||`` evaluated at *sample_size* points per edge --
    deliberately many more than the constraint uses, so the measurement is not
    taken where continuity was imposed.
    """
    if pairs is None:
        pairs = interface_pairs(solid, sample_size=sample_size)

    out: List[np.ndarray] = []
    for pair in pairs:
        surf_a = solid.descendent[pair.face_a].geom
        surf_b = solid.descendent[pair.face_b].geom
        a0, a1 = surf_a.ctp_index
        b0, b1 = surf_b.ctp_index
        va = surface_basis_matrix(surf_a, pair.uv_a) @ control_point_delta[a0:a1]
        vb = surface_basis_matrix(surf_b, pair.uv_b) @ control_point_delta[b0:b1]
        out.append(np.linalg.norm(va - vb, axis=1))
    return np.concatenate(out) if out else np.zeros(0)


# ---------------------------------------------------------------------------
# Constrained solve
# ---------------------------------------------------------------------------

def solve_constrained(
    A: np.ndarray,
    d: np.ndarray,
    C: Optional[np.ndarray] = None,
    smoothing: Optional[np.ndarray] = None,
    smoothing_weight: float = 1e-3,
    ridge: float = 1e-2,
    rank_tol: float = 1e-10,
    max_svd_gb: float = 4.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Solve ``min ||A dP - d||^2 (+ smoothing)`` subject to ``C dP = 0``.

    The constraints are eliminated by solving in an orthonormal basis of
    ``null(C)``, so they hold to machine precision instead of being traded off
    against the data term.  The same null-space basis and factorisation serve
    all three coordinates -- and, for a modal decomposition, every mode.

    *ridge* damps near-singular directions.  ``lstsq`` alone only handles exactly
    singular ones, which lets control points fly off by orders of magnitude while
    barely improving the fit.  Because ``Z`` is orthonormal, a ridge in null-space
    coordinates is exactly a ridge on the control-point displacements.
    """
    A = np.asarray(A, dtype=float)
    d = np.asarray(d, dtype=float)
    n_cols = A.shape[1]

    # Guard the dense SVD below.  On a tall constraint matrix numpy keeps U at
    # (rows, n_cols) and LAPACK wants a further copy or two, so an innocuous
    # looking collocation density can demand many GB and take the machine down
    # rather than raising.  Refuse loudly instead.
    if C is not None and len(C):
        need_gb = 3.0 * len(C) * n_cols * 8 / 2 ** 30
        if need_gb > max_svd_gb:
            raise MemoryError(
                f"constraint SVD needs about {need_gb:.1f} GB "
                f"({len(C)} x {n_cols}), above max_svd_gb={max_svd_gb:.1f}. "
                f"Reduce the collocation sample count: the constraint rank "
                f"saturates well before the memory does."
            )

    if C is not None and len(C):
        # Columns of Z span null(C); rank comes from the singular values.  Ask for
        # the full basis only when C is wide, otherwise U would be (m x m) and the
        # thin form already returns every right singular vector.
        Cm = np.asarray(C, dtype=float)
        _, s, Vt = np.linalg.svd(Cm, full_matrices=Cm.shape[0] < Cm.shape[1])
        rank = int((s > rank_tol * max(1.0, s[0])).sum())
        Z = Vt[rank:].T
    else:
        rank = 0
        Z = np.eye(n_cols)

    if Z.shape[1] == 0:
        raise ValueError("continuity constraints leave no degrees of freedom")

    M = A @ Z
    rhs = d
    if smoothing is not None and len(smoothing):
        S = np.asarray(smoothing, dtype=float) @ Z
        M = np.vstack([M, smoothing_weight * S])
        rhs = np.vstack([rhs, np.zeros((S.shape[0], d.shape[1]))])
    if ridge > 0.0:
        M = np.vstack([M, ridge * np.eye(Z.shape[1])])
        rhs = np.vstack([rhs, np.zeros((Z.shape[1], d.shape[1]))])

    y, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    dP = Z @ y

    info = {
        "n_unknowns": float(n_cols),
        "n_data_rows": float(A.shape[0]),
        "n_constraint_rows": float(0 if C is None else len(C)),
        "constraint_rank": float(rank),
        "free_dof": float(Z.shape[1]),
        "ridge": float(ridge),
        "max_ctp_motion": float(np.linalg.norm(dP, axis=1).max()),
        "data_residual_rms": float(np.sqrt((np.linalg.norm(A @ dP - d, axis=1) ** 2).mean()))
        if A.shape[0] else 0.0,
        "constraint_residual_max": float(np.abs(np.asarray(C) @ dP).max())
        if C is not None and len(C) else 0.0,
    }
    return dP, info


# ---------------------------------------------------------------------------
# High-level driver
# ---------------------------------------------------------------------------

@dataclass
class DeformationResult:
    """Outcome of :func:`deform_solid`."""

    control_point_delta: np.ndarray          # (n_ctp, 3)
    control_points: np.ndarray               # (n_ctp, 3), deformed
    footprints: List[NodeFootprint]
    info: Dict[str, float]

    def node_error(
        self,
        solid: Topology,
        displacement: np.ndarray,
    ) -> np.ndarray:
        """Per-footprint distance between the deformed CAD and the deformed node.

        Evaluated at each node's own ``(u, v)``, so this is an exact paired
        measurement -- no closest-point search, no correspondence guess.
        """
        displacement = np.asarray(displacement, dtype=float)
        errors = np.zeros(len(self.footprints))
        by_face: Dict[int, List[int]] = {}
        for k, fp in enumerate(self.footprints):
            by_face.setdefault(fp.face, []).append(k)

        for face_idx, keys in by_face.items():
            surf = solid.descendent[face_idx].geom
            start, stop = surf.ctp_index
            uv = np.array([[self.footprints[k].u, self.footprints[k].v] for k in keys])
            A = surface_basis_matrix(surf, uv)
            achieved = A @ self.control_point_delta[start:stop]
            wanted = displacement[[self.footprints[k].node for k in keys]]
            errors[keys] = np.linalg.norm(achieved - wanted, axis=1)
        return errors


def deform_solid(
    solid: Topology,
    nodes: np.ndarray,
    displacement: np.ndarray,
    continuity_samples: int = 24,
    smoothing_weight: float = 1e-3,
    ridge: float = 1e-2,
    invert_tol: float = 1e-6,
    footprints: Optional[Sequence[NodeFootprint]] = None,
) -> DeformationResult:
    """Deform *solid*'s control net to follow a mesh displacement field.

    Parameters
    ----------
    solid:
        Root topology node, populated by
        :func:`~cadmap.conversion.occ.add_geom_to_solid`.
    nodes:
        ``(N, 3)`` as-designed mesh nodes, in the CAD's own frame.
    displacement:
        ``(N, 3)`` displacement of each node, in the CAD's frame.  Use
        :func:`rigid_transform` first if the mesh was fitted in another frame.
    continuity_samples:
        Samples per shared edge for the C0 constraints.
    smoothing_weight:
        Weight of the second-difference regulariser.
    """
    nodes = np.asarray(nodes, dtype=float)
    displacement = np.asarray(displacement, dtype=float)
    if nodes.shape != displacement.shape:
        raise ValueError("nodes and displacement must have the same shape")

    prints = list(footprints) if footprints is not None else invert_nodes(
        solid, nodes, tol=invert_tol
    )
    if not prints:
        raise ValueError("no mesh node could be located on the CAD faces")

    A, d = build_data_system(solid, prints, displacement)
    C = build_continuity_rows(solid, sample_size=continuity_samples)
    S = build_smoothing_rows(solid)
    dP, info = solve_constrained(
        A, d, C, S, smoothing_weight=smoothing_weight, ridge=ridge
    )

    ctp0 = np.concatenate([f.geom.init_ctp.reshape(-1, 3) for f in solid.descendent])
    info["n_nodes"] = float(len(nodes))
    info["n_footprints"] = float(len(prints))
    info["nodes_located"] = float(len({fp.node for fp in prints}))
    return DeformationResult(dP, ctp0 + dP, prints, info)


# ---------------------------------------------------------------------------
# Refinement: add degrees of freedom where the NURBS space is too coarse
# ---------------------------------------------------------------------------

def refine_surface(surf, n_u: int = 1, n_v: int = 1) -> None:
    """Insert *n_u* / *n_v* knots into *surf*, in place, preserving its geometry.

    Knot insertion is exact: the surface is unchanged point for point, it simply
    gains control points.  That is what lets a face follow a displacement field
    its original net could not represent -- the faces coming out of a STEP file
    are often a single Bezier span (8x4 control points for degree (7,3)), which
    leaves almost no freedom once continuity is imposed.

    The parameterisation is untouched, so every trim curve and every node
    footprint stays valid.
    """
    if n_u <= 0 and n_v <= 0:
        return

    occ_surf = surf._to_geomsurface()
    tol = 1e-12

    def interior_knots(knots: np.ndarray, degree: int, size: int, count: int) -> List[float]:
        lo, hi = float(knots[degree]), float(knots[size])
        existing = np.unique(knots[(knots > lo + tol) & (knots < hi - tol)])
        # Place new knots at the midpoints of the widest existing spans.
        spans = np.unique(np.concatenate([[lo], existing, [hi]]))
        out: List[float] = []
        for _ in range(count):
            widths = np.diff(spans)
            k = int(np.argmax(widths))
            mid = 0.5 * (spans[k] + spans[k + 1])
            out.append(mid)
            spans = np.insert(spans, k + 1, mid)
        return out

    for knot in interior_knots(surf.knotvector_u, surf.degree_u, surf.size_u, max(n_u, 0)):
        occ_surf.InsertUKnot(knot, 1, tol, False)
    for knot in interior_knots(surf.knotvector_v, surf.degree_v, surf.size_v, max(n_v, 0)):
        occ_surf.InsertVKnot(knot, 1, tol, False)

    _reload_surface_from_occ(surf, occ_surf)


def _reload_surface_from_occ(surf, occ_surf) -> None:
    """Replace *surf*'s control net, weights and knot vectors from an OCC surface."""
    su, sv = occ_surf.NbUPoles(), occ_surf.NbVPoles()
    surf.init_ctp = np.array([
        [occ_surf.Pole(i + 1, j + 1).Coord() for j in range(sv)] for i in range(su)
    ])
    surf.weight = np.array([
        [occ_surf.Weight(i + 1, j + 1) for j in range(sv)] for i in range(su)
    ])

    u_min, u_max, v_min, v_max = surf.param_bounds

    def knot_vector(n_knots, knot, mult, lo, hi):
        knots = [knot(i) for i in range(1, n_knots + 1)]
        mults = [mult(i) for i in range(1, n_knots + 1)]
        kv = np.array([k for k, m in zip(knots, mults) for _ in range(m)])
        return (kv - lo) / (hi - lo)

    # The OCC surface was built from the already-normalised knot vectors, so its
    # own bounds are the normalised ones.
    surf.knotvector_u = knot_vector(
        occ_surf.NbUKnots(), occ_surf.UKnot, occ_surf.UMultiplicity, 0.0, 1.0
    )
    surf.knotvector_v = knot_vector(
        occ_surf.NbVKnots(), occ_surf.VKnot, occ_surf.VMultiplicity, 0.0, 1.0
    )
    surf.size_u, surf.size_v = su, sv
    surf.degree_u, surf.degree_v = occ_surf.UDegree(), occ_surf.VDegree()
    surf.DiffSurf = None
    surf.inside_evalpts = None


def reindex_control_points(solid: Topology) -> int:
    """Recompute every face's ``ctp_index`` after refinement.  Returns the total."""
    offset = 0
    for face in solid.descendent:
        surf = face.geom
        count = surf.size_u * surf.size_v
        surf.ctp_index = (offset, offset + count)
        offset += count
    return offset


def refine_solid(
    solid: Topology,
    n_u: int = 1,
    n_v: int = 1,
    faces: Optional[Sequence[int]] = None,
) -> int:
    """Refine every face (or just *faces*) and reindex the global control array."""
    targets = range(len(solid.descendent)) if faces is None else faces
    for i in targets:
        refine_surface(solid.descendent[i].geom, n_u, n_v)
    return reindex_control_points(solid)


# ---------------------------------------------------------------------------
# Dense field sampling: the displacement is known everywhere on the mesh
# ---------------------------------------------------------------------------

def _closest_point_on_triangles(
    points: np.ndarray,
    tri: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Closest point and barycentric weights for each ``points[i]`` on ``tri[i]``.

    Vectorised point-triangle projection: solve in the triangle plane, then clamp
    into the triangle and renormalise the weights.
    """
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac, ap = b - a, c - a, points - a

    d00 = np.einsum("ij,ij->i", ab, ab)
    d01 = np.einsum("ij,ij->i", ab, ac)
    d11 = np.einsum("ij,ij->i", ac, ac)
    d20 = np.einsum("ij,ij->i", ap, ab)
    d21 = np.einsum("ij,ij->i", ap, ac)

    denom = d00 * d11 - d01 * d01
    safe = np.where(np.abs(denom) < 1e-300, 1.0, denom)
    v = (d11 * d20 - d01 * d21) / safe
    w = (d00 * d21 - d01 * d20) / safe

    v = np.clip(v, 0.0, 1.0)
    w = np.clip(w, 0.0, 1.0)
    over = v + w > 1.0
    total = np.where(over, v + w, 1.0)
    v = np.where(over, v / total, v)
    w = np.where(over, w / total, w)
    u = 1.0 - v - w

    bary = np.stack([u, v, w], axis=1)
    closest = a + v[:, None] * ab + w[:, None] * ac
    return closest, bary


class MeshField:
    """Piecewise-linear displacement field carried by a triangle mesh.

    The nodes are not the only place the field is known: it is defined across
    every triangle by linear interpolation.  Sampling it lets each CAD face
    receive data in proportion to its area rather than to wherever the mesher
    happened to put nodes -- on a real part some faces carry a thousand nodes and
    others three.
    """

    def __init__(self, nodes: np.ndarray, triangles: np.ndarray, displacement: np.ndarray):
        from scipy.spatial import cKDTree

        self.nodes = np.asarray(nodes, dtype=float)
        self.triangles = np.asarray(triangles, dtype=np.int64)
        self.displacement = np.asarray(displacement, dtype=float)
        self.tri = self.nodes[self.triangles]
        self.tree = cKDTree(self.tri.mean(axis=1))

    def locate(self, points: np.ndarray, candidates: int = 12):
        """Return ``(triangle_index, barycentric, distance)`` for each point.

        This is the projection ``Pi_h`` onto the mesh together with the three
        nonzero P1 shape-function values there, which is what a variational
        assembly needs in order to build the mesh-to-CAD coupling matrix rather
        than only an interpolated value.
        """
        points = np.atleast_2d(np.asarray(points, dtype=float))
        k = min(candidates, len(self.tri))
        _, idx = self.tree.query(points, k=k)
        idx = np.atleast_2d(idx)

        best_d = np.full(len(points), np.inf)
        best_tri = np.zeros(len(points), dtype=np.int64)
        best_bary = np.zeros((len(points), 3))
        for col in range(idx.shape[1]):
            tri_idx = idx[:, col]
            closest, bary = _closest_point_on_triangles(points, self.tri[tri_idx])
            dist = np.linalg.norm(closest - points, axis=1)
            better = dist < best_d
            if not better.any():
                continue
            best_tri[better] = tri_idx[better]
            best_bary[better] = bary[better]
            best_d[better] = dist[better]
        return best_tri, best_bary, best_d

    def interpolate(self, points: np.ndarray, candidates: int = 12):
        """Return ``(displacement, distance)`` interpolated at *points*."""
        tri_idx, bary, dist = self.locate(points, candidates)
        nodal = self.displacement[self.triangles[tri_idx]]
        return np.einsum("nb,nbc->nc", bary, nodal), dist


def _trim_classifier(face):
    from OCC.Core.BRepTopAdaptor import BRepTopAdaptor_FClass2d

    from cadmap.conversion.export import build_occ_face

    return BRepTopAdaptor_FClass2d(build_occ_face(face), 1e-7)


def _grid_inside_trim(surf, classifier, per_ctp: float, min_side: int, max_side: int):
    """Parametric grid points of *surf* that fall inside its trim boundary."""
    from OCC.Core.gp import gp_Pnt2d
    from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON

    n_u = int(np.clip(round(per_ctp * surf.size_u), min_side, max_side))
    n_v = int(np.clip(round(per_ctp * surf.size_v), min_side, max_side))
    uu, vv = np.meshgrid(
        np.linspace(0.0, 1.0, n_u), np.linspace(0.0, 1.0, n_v), indexing="ij"
    )
    uv = np.stack([uu.ravel(), vv.ravel()], axis=1)
    inside = np.array([
        classifier.Perform(gp_Pnt2d(float(u), float(v))) in (TopAbs_IN, TopAbs_ON)
        for u, v in uv
    ])
    return uv[inside]


def sample_faces(
    solid: Topology,
    field: MeshField,
    per_ctp: float = 2.0,
    min_side: int = 6,
    max_side: int = 40,
):
    """Build the data system from the field sampled on every face trim region.

    Returns ``(A, d, face_of_row)``, sized so each face gets about *per_ctp*
    samples per control point in each direction.
    """
    n_cols = control_point_count(solid)
    blocks, values, owners = [], [], []

    for face_idx, face in enumerate(solid.descendent):
        surf = face.geom
        uv = _grid_inside_trim(surf, _trim_classifier(face), per_ctp, min_side, max_side)
        if not len(uv):
            continue

        points = np.asarray(surf.evaluate_list(surf.init_ctp, uv.tolist()), dtype=float)
        disp, _ = field.interpolate(points)

        block = np.zeros((len(uv), n_cols))
        start, stop = surf.ctp_index
        block[:, start:stop] = surface_basis_matrix(surf, uv)
        blocks.append(block)
        values.append(disp)
        owners.append(np.full(len(uv), face_idx))

    if not blocks:
        return np.zeros((0, n_cols)), np.zeros((0, 3)), np.zeros(0, dtype=int)
    return np.vstack(blocks), np.vstack(values), np.concatenate(owners)


# ---------------------------------------------------------------------------
# Adaptive refinement driven by per-face residual
# ---------------------------------------------------------------------------

def face_representation_residual(face, field: MeshField, per_ctp: float = 2.0) -> float:
    """Max residual of fitting *field* on one face alone, with no constraints.

    Measures representability only: how well this face's NURBS space can express
    the field, independent of how the global solve distributes freedom.
    """
    surf = face.geom
    uv = _grid_inside_trim(surf, _trim_classifier(face), per_ctp, 6, 40)
    if len(uv) < 4:
        return 0.0
    points = np.asarray(surf.evaluate_list(surf.init_ctp, uv.tolist()), dtype=float)
    disp, _ = field.interpolate(points)
    A = surface_basis_matrix(surf, uv)
    x, *_ = np.linalg.lstsq(A, disp, rcond=None)
    return float(np.linalg.norm(A @ x - disp, axis=1).max())


def refine_solid_adaptive(
    solid: Topology,
    field: MeshField,
    tolerance: float,
    max_rounds: int = 6,
    max_ctp_side: int = 12,
    per_ctp: float = 2.0,
    verbose: bool = False,
) -> Dict[int, Tuple[int, int]]:
    """Refine each face until it can represent the field to *tolerance*.

    Faces converted from analytic surfaces arrive with almost no freedom -- a
    plane becomes a 2x2 bilinear patch that can only deform affinely, a cylinder
    3x2 -- while B-spline faces arrive far richer.  Uniform refinement therefore
    spends DOF on faces that are already adequate and still starves the ones that
    carry the error, so refine per face against its own residual.

    Returns the final ``(size_u, size_v)`` of every face that was refined.
    """
    changed: Dict[int, Tuple[int, int]] = {}

    for face_idx, face in enumerate(solid.descendent):
        surf = face.geom
        for _ in range(max_rounds):
            residual = face_representation_residual(face, field, per_ctp)
            if residual <= tolerance:
                break
            add_u = 1 if surf.size_u < max_ctp_side else 0
            add_v = 1 if surf.size_v < max_ctp_side else 0
            if not (add_u or add_v):
                break
            refine_surface(surf, add_u, add_v)
            changed[face_idx] = (surf.size_u, surf.size_v)
            if verbose:
                print(f"    face {face_idx}: residual {residual:.4f} -> "
                      f"net {surf.size_u}x{surf.size_v}")

    reindex_control_points(solid)
    return changed


def continuity_samples_for(solid: Topology, per_ctp: float = 3.0, minimum: int = 16) -> int:
    """Collocation points per shared edge, scaled to the finest control net.

    Continuity is imposed at samples, so a finer net needs more of them: at a
    fixed count the surfaces gain freedom to separate *between* collocation
    points, which is how refinement breaks watertightness.
    """
    widest = max(max(f.geom.size_u, f.geom.size_v) for f in solid.descendent)
    return int(max(minimum, round(per_ctp * widest)))
