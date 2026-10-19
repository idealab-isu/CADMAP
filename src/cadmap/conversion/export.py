"""Geometry-to-OCC export: rebuilds a B-Rep solid from the cadmap NURBS classes.

This is the inverse of :mod:`cadmap.conversion.occ`.  Given a
:class:`~cadmap.geometry.topology.Topology` tree whose nodes carry
``BsplineSurface`` / ``Curve2d`` geometry (and, optionally, a flat array of
updated control points produced by the optimizer), it rebuilds the trimmed
faces, sews them into a closed shell and writes a STEP file.

    solid = step_to_solid("part.step")
    shape = solid_to_occ_shape(solid)          # or solid_to_occ_shape(solid, ctp)
    write_step(shape, "part_roundtrip.step")
"""

from __future__ import annotations

import warnings
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepCheck import BRepCheck_NoError, BRepCheck_Shell
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeSolid,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_Sewing,
)
from OCC.Core.BRepLib import breplib
from OCC.Core.BRepTools import breptools
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Shape, TopoDS_Shell, TopoDS_Solid, topods
from OCC.Core.TopTools import TopTools_ListOfShape

from cadmap.geometry.topology import Topology

__all__ = [
    "face_control_points",
    "build_occ_face",
    "build_occ_faces",
    "sew_faces",
    "is_closed_shell",
    "assemble_shape",
    "solid_to_occ_shape",
    "write_step",
]

DEFAULT_SEW_TOL = 1e-6
_FACE_TOL = 1e-7

#: Tolerance for rebuilding an edge's 3-D curve from its pcurve.  It bounds how
#: far a rebuilt boundary can drift from the true one, so it must stay well
#: below the sewing tolerance or adjacent faces will not meet.
_BUILD_CURVES_3D_TOL = 1e-9


# ---------------------------------------------------------------------------
# Control points
# ---------------------------------------------------------------------------

def face_control_points(surf: Any, ctpall_new: Optional[Any] = None) -> np.ndarray:
    """Return the ``(size_u, size_v, 3)`` control net of *surf*.

    *ctpall_new* is the flat ``(N, 3)`` control-point array (numpy or torch)
    shared by every face of the solid; the slice belonging to *surf* is taken
    from ``surf.ctp_index``.  When it is ``None`` the surface's own
    ``init_ctp`` is used, i.e. the as-designed geometry.
    """
    if ctpall_new is None:
        ctp = surf.init_ctp
    else:
        start, stop = surf.ctp_index
        ctp = ctpall_new[start:stop]
        if hasattr(ctp, "detach"):  # torch tensor
            ctp = ctp.detach().cpu().numpy()
    # float64: OCC's gp_Pnt rejects numpy float32, which is what the optimizer
    # hands back.
    return np.asarray(ctp, dtype=np.float64).reshape(surf.size_u, surf.size_v, 3)


# ---------------------------------------------------------------------------
# Face reconstruction
# ---------------------------------------------------------------------------

def build_occ_face(face_topo: Topology, ctpall_new: Optional[Any] = None) -> TopoDS_Face:
    """Rebuild one trimmed ``TopoDS_Face`` from a face node of the topology tree.

    The untrimmed surface comes from ``face_topo.geom``; every wire is rebuilt
    from the 2-D trim curves stored on its edges, so the parametric trim
    structure of the original STEP file is preserved.  The orientation of the
    original face is re-applied so the rebuilt shell keeps a consistent
    outward normal.
    """
    surf = face_topo.geom
    if surf is None:
        raise ValueError(f"Face {face_topo} has no geometry - run add_geom_to_solid first.")

    occ_surf = surf._to_geomsurface(face_control_points(surf, ctpall_new))

    outer_wires: List = []
    inner_wires: List = []
    for wire_topo in face_topo.descendent:
        wire_builder = BRepBuilderAPI_MakeWire()
        wire_edges = TopTools_ListOfShape()
        for edge_topo in wire_topo.descendent:
            if edge_topo.geom2d is None:
                raise ValueError(f"Edge {edge_topo} has no 2-D trim curve.")
            trim2d = edge_topo.geom2d._to_Geom2d_BSplineCurve()
            edge_maker = BRepBuilderAPI_MakeEdge(trim2d, occ_surf)
            if not edge_maker.IsDone():
                raise RuntimeError(f"Failed to rebuild edge {edge_topo}.")
            occ_edge = edge_maker.Edge()
            breplib.BuildCurves3d(occ_edge, _BUILD_CURVES_3D_TOL)
            wire_edges.Append(occ_edge)

        # TopExp traversal is not necessarily connection order. Adding edges
        # individually can silently discard a disconnected intermediate edge
        # and later report IsDone() for only a subset of the original wire.
        wire_builder.Add(wire_edges)

        if not wire_builder.IsDone():
            raise RuntimeError(f"Failed to rebuild wire {wire_topo}.")
        occ_wire = wire_builder.Wire()
        explorer = TopExp_Explorer(occ_wire, TopAbs_EDGE)
        edge_count = 0
        while explorer.More():
            edge_count += 1
            explorer.Next()
        if edge_count != len(wire_topo.descendent):
            raise RuntimeError(f"Rebuilding wire {wire_topo} lost boundary edges.")

        if breptools.OuterWire(face_topo.TopoDS).IsSame(wire_topo.TopoDS):
            outer_wires.append(occ_wire)
        else:
            inner_wires.append(occ_wire)

    if not outer_wires:
        face_builder = BRepBuilderAPI_MakeFace(occ_surf, _FACE_TOL)
    elif len(outer_wires) == 1:
        face_builder = BRepBuilderAPI_MakeFace(occ_surf, outer_wires[0], True)
    else:
        raise ValueError(f"Face {face_topo} has more than one outer wire.")
    for wire in inner_wires:
        face_builder.Add(wire)

    if not face_builder.IsDone():
        raise RuntimeError(f"Failed to rebuild face {face_topo}.")

    occ_face = face_builder.Face()
    occ_face.Orientation(face_topo.TopoDS.Orientation())
    return occ_face


def build_occ_faces(solid: Topology, ctpall_new: Optional[Any] = None) -> List[TopoDS_Face]:
    """Rebuild every trimmed face of *solid*, in topology-tree order."""
    return [build_occ_face(face, ctpall_new) for face in solid.descendent]


# ---------------------------------------------------------------------------
# Shell / solid assembly
# ---------------------------------------------------------------------------

def sew_faces(
    faces: Sequence[TopoDS_Face],
    tol: float = DEFAULT_SEW_TOL,
) -> Tuple[TopoDS_Shape, List[TopoDS_Shell]]:
    """Sew *faces* into shells, merging the edges they share.

    Returns the sewed shape together with the list of shells it contains.  A
    watertight result is one shell for which :func:`is_closed_shell` holds;
    more than one shell (or an open one) means adjacent faces did not meet
    within *tol*, which is the signal that the deformation opened a gap.
    """
    sewing = BRepBuilderAPI_Sewing(tol)
    for f in faces:
        sewing.Add(f)
    sewing.Perform()
    sewed = sewing.SewedShape()

    shells: List[TopoDS_Shell] = []
    explorer = TopExp_Explorer(sewed, TopAbs_SHELL)
    while explorer.More():
        shells.append(topods.Shell(explorer.Current()))
        explorer.Next()

    if not shells:
        # A single face that closes on itself (a whole sphere or torus) is
        # returned as the bare face; wrap it so it can still form a solid.
        shells = [_shell_of_faces(sewed)]
    return sewed, shells


def _shell_of_faces(shape: TopoDS_Shape) -> TopoDS_Shell:
    """Collect every face of *shape* into one shell."""
    builder = BRep_Builder()
    shell = TopoDS_Shell()
    builder.MakeShell(shell)
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        builder.Add(shell, topods.Face(explorer.Current()))
        explorer.Next()
    return shell


def is_closed_shell(shell: TopoDS_Shell) -> bool:
    """True when *shell* bounds a volume without gaps.

    ``TopoDS_Shell.Closed()`` only reports a cached flag, which sewing leaves
    unset on shells it did not build itself, so check the topology instead.
    """
    return BRepCheck_Shell(shell).Closed() == BRepCheck_NoError


def assemble_shape(
    faces: Sequence[TopoDS_Face],
    sew: bool = True,
    tol: float = DEFAULT_SEW_TOL,
) -> TopoDS_Shape:
    """Assemble rebuilt *faces* into a solid (see :func:`solid_to_occ_shape`)."""
    if not sew:
        builder = BRep_Builder()
        occ_shell = TopoDS_Shell()
        builder.MakeShell(occ_shell)
        for f in faces:
            builder.Add(occ_shell, f)
        occ_solid = TopoDS_Solid()
        builder.MakeSolid(occ_solid)
        builder.Add(occ_solid, occ_shell)
        return occ_solid

    sewed, shells = sew_faces(faces, tol)
    if not shells or not all(is_closed_shell(shell) for shell in shells):
        warnings.warn(
            f"Sewing {len(faces)} faces at tolerance {tol:g} produced "
            f"{len(shells)} shell(s), not all closed: the exported shape is not "
            f"watertight.  Adjacent faces may have drifted apart during fitting, "
            f"or the tolerance may be too tight for this model's units.",
            RuntimeWarning,
            stacklevel=2,
        )
        return sewed

    solid_builder = BRepBuilderAPI_MakeSolid()
    for shell in shells:
        solid_builder.Add(shell)
    if not solid_builder.IsDone():
        return sewed
    return solid_builder.Solid()


def solid_to_occ_shape(
    solid: Topology,
    ctpall_new: Optional[Any] = None,
    sew: bool = True,
    tol: float = DEFAULT_SEW_TOL,
) -> TopoDS_Shape:
    """Rebuild an OCC shape from the cadmap topology and (optionally) new control points.

    Parameters
    ----------
    solid:
        Root topology node, already populated by
        :func:`~cadmap.conversion.occ.add_geom_to_solid`.
    ctpall_new:
        Flat ``(N, 3)`` control-point array (numpy or torch) from the
        optimizer.  ``None`` exports the as-designed geometry.
    sew:
        Sew the faces so adjacent ones share edges.  This is what makes the
        exported STEP a closed, positive-volume solid; without it every face
        lands in its own shell.
    tol:
        Sewing tolerance in model units.

    Returns
    -------
    A ``TopoDS_Solid`` when the faces sew into closed shells, otherwise the
    sewed (open) shape - still writable to STEP, just not watertight.
    """
    return assemble_shape(build_occ_faces(solid, ctpall_new), sew=sew, tol=tol)


# ---------------------------------------------------------------------------
# STEP output
# ---------------------------------------------------------------------------

def write_step(occ_shape: TopoDS_Shape, filename: str, schema: str = "AP214") -> None:
    """Write *occ_shape* to a STEP file."""
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.Interface import Interface_Static
    from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer

    writer = STEPControl_Writer()
    Interface_Static.SetCVal("write.step.schema", schema)
    writer.Transfer(occ_shape, STEPControl_AsIs)
    if writer.Write(str(filename)) != IFSelect_RetDone:
        raise RuntimeError(f"Failed to write STEP file: {filename}")
