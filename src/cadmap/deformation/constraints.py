"""Build and manage the NURBS continuity constraint matrix for topology-preserving deformation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from geomdl.helpers import basis_function, find_span_binsearch
from scipy.linalg import orth

from cadmap.geometry.topology import Topology


# ---------------------------------------------------------------------------
# Topology traversal helpers
# ---------------------------------------------------------------------------

def get_all_edges3d(solid: Topology) -> List[Any]:
    """Return a deduplicated list of all ``Curve3d`` geometry objects in *solid*."""
    seen = []
    for face in solid.descendent:
        for wire in face.descendent:
            for edge in wire.descendent:
                if not any(edge.geom.IsSame(e) for e in seen):
                    seen.append(edge.geom)
    return seen


def search_Surf3dEdge2d_of_edge3d(edge3d, solid: Topology) -> List[Tuple[Any, Any]]:
    """For *edge3d* return all ``(BsplineSurface, Curve2d)`` pairs that contain it."""
    result = []
    for face in solid.descendent:
        for wire in face.descendent:
            for edge in wire.descendent:
                if edge.geom.IsSame(edge3d):
                    result.append((face.geom, edge.geom2d))
    return result


def get_unique_vertices(solid: Topology) -> List:
    """Return a list of unique ``TopoDS_Vertex`` shapes from *solid*."""
    from OCC.Core.TopExp import topexp, TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_VERTEX
    from OCC.Core.TopoDS import topods
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape

    vmap = TopTools_IndexedMapOfShape()
    topexp.MapShapes(solid.TopoDS, TopAbs_VERTEX, vmap)
    return [topods.Vertex(vmap.FindKey(i)) for i in range(1, vmap.Size() + 1)]


def get_connectivity_of_vertex(vertex, solid: Topology) -> Tuple[List[Topology], List[Topology]]:
    """Return ``(connected_faces, connected_edges)`` for *vertex* in *solid*."""
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_VERTEX

    connected_faces = []
    connected_edges = []

    for face in solid.descendent:
        face_has_vertex = False
        for wire in face.descendent:
            for edge in wire.descendent:
                exp_v = TopExp_Explorer(edge.TopoDS, TopAbs_VERTEX)
                edge_has_v = False
                while exp_v.More():
                    if exp_v.Current().IsSame(vertex):
                        edge_has_v = True
                        break
                    exp_v.Next()
                if edge_has_v:
                    face_has_vertex = True
                    if not any(e.TopoDS.IsSame(edge.TopoDS) for e in connected_edges):
                        connected_edges.append(edge)
        if face_has_vertex:
            connected_faces.append(face)

    return connected_faces, connected_edges


def get_edges_of_faces(faces: List[Topology], exclude_edges: List[Topology] = None) -> List[Topology]:
    """Return all edges belonging to *faces*, excluding those in *exclude_edges*."""
    if exclude_edges is None:
        exclude_edges = []
    result = []
    for face in faces:
        for wire in face.descendent:
            for edge in wire.descendent:
                if any(e.TopoDS.IsSame(edge.TopoDS) for e in exclude_edges):
                    continue
                if not any(e.TopoDS.IsSame(edge.TopoDS) for e in result):
                    result.append(edge)
    return result


def get_cp_indices_of_faces(faces: List[Topology]) -> List[int]:
    """Return all global control-point indices belonging to *faces*."""
    indices = []
    for face in faces:
        start, end = face.geom.ctp_index
        indices.extend(range(start, end))
    return list(set(indices))


def get_constraint_rows_for_edges(
    edges: List[Topology],
    edge_map: Dict,
    periodic_map: Dict,
    faces: List[Topology],
) -> List[int]:
    """Collect row indices from ``all_basis`` for *edges* and their periodic surfaces."""
    rows = []
    for edge in edges:
        if hasattr(edge, "geom") and edge.geom in edge_map:
            rows.extend(edge_map[edge.geom])
        else:
            for key, val in edge_map.items():
                if hasattr(edge, "geom") and key.curv_id == edge.geom.curv_id:
                    rows.extend(val)
                    break
    for face in faces:
        if face.geom in periodic_map:
            rows.extend(periodic_map[face.geom])
    return sorted(set(rows))


# ---------------------------------------------------------------------------
# Constraint-matrix construction
# ---------------------------------------------------------------------------

def build_constraint_matrix(
    solid: Topology,
    ctpall: np.ndarray,
    sample_size: int = 15,
) -> Tuple[np.ndarray, Dict, Dict]:
    """Assemble the full continuity constraint matrix for *solid*.

    Each row encodes that two adjacent surfaces must evaluate to the same 3-D
    point along their shared edge.  Also appends periodic surface constraints.

    Parameters
    ----------
    solid:
        Root :class:`~cadmap.geometry.topology.Topology` node (faces as descendants).
    ctpall:
        ``(K, 3)`` stacked control-point array for the whole solid.
    sample_size:
        Number of sample points per edge used to build constraints.

    Returns
    -------
    all_basis:
        ``(R, K)`` constraint matrix.
    edge_to_constraint_rows:
        Mapping ``Curve3d → List[int]`` of row indices in *all_basis*.
    periodic_constraints_map:
        Mapping ``BsplineSurface → List[int]`` of periodic-constraint rows.
    """
    from cadmap.deformation.constraints import (
        get_all_edges3d,
        search_Surf3dEdge2d_of_edge3d,
        OCC_search_pair_points,
    )

    all_edge3d = get_all_edges3d(solid)
    n_ctp = ctpall.shape[0]
    all_basis = np.zeros((sample_size * len(all_edge3d), n_ctp))
    edge_to_constraint_rows: Dict = {}

    for edge in all_edge3d:
        edge_idx = int(edge.curv_id.split("_")[-1])
        row_range = list(range(edge_idx * sample_size, (edge_idx + 1) * sample_size))
        edge_to_constraint_rows[edge] = row_range

        coef_sign = 1
        surf3d_edge2d_pairs = search_Surf3dEdge2d_of_edge3d(edge, solid)

        uv_list_3d = None
        for on_surf3d, on_edge2d in surf3d_edge2d_pairs:
            edge2d_geomdl = on_edge2d._to_geomdl()
            edge2d_geomdl.sample_size = sample_size

            if coef_sign == 1:
                uv_list = np.clip(np.array(edge2d_geomdl.evalpts), 0, 1)
                uv_list_3d = on_surf3d.evaluate_list(on_surf3d.init_ctp, uv_list)
            else:
                uv_list = [
                    OCC_search_pair_points(on_surf3d, loc, on_edge2d)
                    for loc in uv_list_3d
                ]
                uv_list = np.clip(np.array(uv_list), 0, 1)

            for idx, uv in enumerate(uv_list):
                u_span = find_span_binsearch(
                    on_surf3d.degree_u, on_surf3d.knotvector_u, on_surf3d.size_u, uv[0]
                )
                v_span = find_span_binsearch(
                    on_surf3d.degree_v, on_surf3d.knotvector_v, on_surf3d.size_v, uv[1]
                )
                Nu = np.zeros((on_surf3d.size_u, 1))
                Nv = np.zeros((on_surf3d.size_v, 1))
                Nu_cp = np.array(basis_function(on_surf3d.degree_u, on_surf3d.knotvector_u, u_span, uv[0]))
                Nv_cp = np.array(basis_function(on_surf3d.degree_v, on_surf3d.knotvector_v, v_span, uv[1]))
                Nu[u_span - on_surf3d.degree_u: u_span + 1] = Nu_cp[:, np.newaxis]
                Nv[v_span - on_surf3d.degree_v: v_span + 1] = Nv_cp[:, np.newaxis]
                NuNvW = Nu @ Nv.T * on_surf3d.weight
                Rational_NuNvW = NuNvW / NuNvW.sum()
                row = idx + edge_idx * sample_size
                all_basis[row, on_surf3d.ctp_index[0]: on_surf3d.ctp_index[1]] = (
                    Rational_NuNvW.reshape(-1) * coef_sign
                )

            coef_sign *= -1

    # --- periodic surface constraints ---
    periodic_constraints_map: Dict = {}
    for surface_topo in solid.descendent:
        surf = surface_topo.geom
        current_rows: List[int] = []

        if surf.periodic_u:
            new_c = np.zeros((surf.size_v, n_ctp))
            mask = np.zeros((surf.size_v, surf.size_u, surf.size_v))
            np.fill_diagonal(mask[:, 0, :], 1)
            np.fill_diagonal(mask[:, -1, :], -1)
            new_c[:, surf.ctp_index[0]: surf.ctp_index[1]] = mask.reshape(surf.size_v, -1)
            start = all_basis.shape[0]
            all_basis = np.concatenate([all_basis, new_c])
            current_rows.extend(range(start, all_basis.shape[0]))

        if surf.periodic_v:
            new_c = np.zeros((surf.size_u, n_ctp))
            mask = np.zeros((surf.size_u, surf.size_u, surf.size_v))
            np.fill_diagonal(mask[:, :, 0], 1)
            np.fill_diagonal(mask[:, :, -1], -1)
            new_c[:, surf.ctp_index[0]: surf.ctp_index[1]] = mask.reshape(surf.size_u, -1)
            start = all_basis.shape[0]
            all_basis = np.concatenate([all_basis, new_c])
            current_rows.extend(range(start, all_basis.shape[0]))

        periodic_constraints_map[surf] = current_rows

    return all_basis, edge_to_constraint_rows, periodic_constraints_map


def compute_local_basis(
    all_basis: np.ndarray,
    row_indices: List[int],
    rcond: float = 1e-5,
) -> np.ndarray:
    """Return the orthonormal row-space basis of ``all_basis[row_indices, :]``."""
    if len(row_indices) == 0:
        return np.zeros((0, all_basis.shape[1]))
    local_A = all_basis[row_indices, :]
    return orth(local_A.T, rcond=rcond).T


# ---------------------------------------------------------------------------
# OCC helper for finding paired UV on adjacent surfaces
# ---------------------------------------------------------------------------

def OCC_search_pair_points(on_surf3d, loc_3d, on_edge2d) -> List[float]:
    """Find the (u, v) on *on_surf3d* corresponding to the 3-D point *loc_3d* on *on_edge2d*."""
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeVertex
    from OCC.Core.BRepLib import breplib
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve
    from OCC.Core.gp import gp_Pnt

    occ_surface = on_surf3d._to_geomsurface()
    occ_curve2d = on_edge2d._to_Geom2d_BSplineCurve()
    edge_maker = BRepBuilderAPI_MakeEdge(occ_curve2d, occ_surface)
    if not edge_maker.IsDone():
        raise RuntimeError("Could not build edge for pair-point search.")
    topo_edge = edge_maker.Edge()
    breplib.BuildCurves3d(topo_edge, 1e-12)

    gp_p3d = gp_Pnt(*loc_3d)
    curve_handle, _, _ = BRep_Tool.Curve(topo_edge)
    proj = GeomAPI_ProjectPointOnCurve(gp_p3d, curve_handle)
    if proj.NbPoints() > 0:
        curve_u = proj.LowerDistanceParameter()
    else:
        d_start = gp_p3d.Distance(curve_handle.Value(curve_handle.FirstParameter()))
        d_end = gp_p3d.Distance(curve_handle.Value(curve_handle.LastParameter()))
        curve_u = curve_handle.FirstParameter() if d_start < d_end else curve_handle.LastParameter()

    uv = occ_curve2d.Value(curve_u).Coord()
    return list(uv)
