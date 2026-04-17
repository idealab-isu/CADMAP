"""OCC topology-to-geometry conversion: reads a B-Rep solid and populates
the Topology tree with BsplineSurface / Curve3d / Curve2d geometry objects.

The public entry point is :class:`SolidConvertor`.  Call ``convert()`` after
constructing it with the root :class:`~cadmap.geometry.topology.Topology` node.

    solid = Topology(read_step_file(path))
    SolidConvertor(solid).convert()
    add_geom_to_solid(solid)
    solid_to_device(solid, "cpu")
    create_diffsurf_of_solid(solid)
"""

from __future__ import annotations

import copy
from typing import Any, List, Optional, Tuple

import numpy as np

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Curve2d, BRepAdaptor_Surface
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_NurbsConvert,
    BRepBuilderAPI_MakeWire,
)
from OCC.Core.BRepLib import breplib
from OCC.Core.BRepTools import breptools
from OCC.Core.Geom import Geom_BSplineCurve, Geom_BSplineSurface
from OCC.Core.Geom2d import Geom2d_BSplineCurve
from OCC.Core.GeomAbs import (
    GeomAbs_BSplineCurve,
    GeomAbs_BSplineSurface,
    GeomAbs_BezierCurve,
    GeomAbs_Circle,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_Line,
    GeomAbs_OtherCurve,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_Torus,
)
from OCC.Core.GeomConvert import geomconvert
from OCC.Core.gp import gp_Pnt, gp_Pnt2d
from OCC.Core.TColgp import TColgp_Array1OfPnt, TColgp_Array1OfPnt2d, TColgp_Array2OfPnt
from OCC.Core.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal, TColStd_Array2OfReal
from OCC.Core.TopAbs import (
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_REVERSED,
    TopAbs_VERTEX,
    TopAbs_WIRE,
)
from OCC.Core.TopoDS import topods, TopoDS_Face
from OCC.Core.TopExp import TopExp_Explorer

from cadmap.geometry.nurbs import BsplineSurface, Curve2d, Curve3d
from cadmap.geometry.topology import Topology

# ---------------------------------------------------------------------------
# Helper: STEP file reader (re-exported for convenience)
# ---------------------------------------------------------------------------

from OCC.Extend.DataExchange import read_step_file  # noqa: F401


# ---------------------------------------------------------------------------
# Low-level OCC curve/surface helpers  (ported from myclass.py)
# ---------------------------------------------------------------------------

def _convert_line_to_bspline3d(adaptor: BRepAdaptor_Curve) -> Geom_BSplineCurve:
    p1 = adaptor.Value(adaptor.FirstParameter())
    p2 = adaptor.Value(adaptor.LastParameter())
    pts = TColgp_Array1OfPnt(1, 2)
    pts.SetValue(1, p1)
    pts.SetValue(2, p2)
    knots = TColStd_Array1OfReal(1, 2)
    mults = TColStd_Array1OfInteger(1, 2)
    knots.SetValue(1, 0.0); knots.SetValue(2, 1.0)
    mults.SetValue(1, 2); mults.SetValue(2, 2)
    return Geom_BSplineCurve(pts, knots, mults, 1)


def _convert_line_to_bspline2d(adaptor: BRepAdaptor_Curve2d) -> Geom2d_BSplineCurve:
    p1 = adaptor.Value(adaptor.FirstParameter())
    p2 = adaptor.Value(adaptor.LastParameter())
    pts = TColgp_Array1OfPnt2d(1, 2)
    pts.SetValue(1, p1)
    pts.SetValue(2, p2)
    knots = TColStd_Array1OfReal(1, 2)
    mults = TColStd_Array1OfInteger(1, 2)
    knots.SetValue(1, 0.0); knots.SetValue(2, 1.0)
    mults.SetValue(1, 2); mults.SetValue(2, 2)
    return Geom2d_BSplineCurve(pts, knots, mults, 1)


def _curve3d_adaptor_to_bspline(adaptor: BRepAdaptor_Curve) -> Geom_BSplineCurve:
    t = adaptor.GetType()
    if t == GeomAbs_Line:
        return _convert_line_to_bspline3d(adaptor)
    if t == GeomAbs_Circle:
        edge_maker = BRepBuilderAPI_MakeEdge(adaptor.Circle())
        if not edge_maker.IsDone():
            raise RuntimeError("Circle edge creation failed.")
        nurbs_conv = BRepBuilderAPI_NurbsConvert(edge_maker.Edge())
        nurbs_conv.Perform(edge_maker.Edge())
        if not nurbs_conv.IsDone():
            raise RuntimeError("Circle NURBS conversion failed.")
        return BRepAdaptor_Curve(topods.Edge(nurbs_conv.Shape())).BSpline()
    if t == GeomAbs_BSplineCurve:
        return adaptor.BSpline()
    if t == GeomAbs_BezierCurve:
        return Geom_BSplineCurve(adaptor.Bezier())
    raise NotImplementedError(f"Unsupported 3-D curve type: {t}")


def _extract_curve_data(bspline) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return (control_points, knot_vector, weights, degree) from an OCC BSpline curve."""
    degree = bspline.Degree()
    poles = [bspline.Pole(i).Coord() for i in range(1, bspline.NbPoles() + 1)]
    control_points = np.array(poles)
    knots = [bspline.Knot(i) for i in range(1, bspline.NbKnots() + 1)]
    mults = [bspline.Multiplicity(i) for i in range(1, bspline.NbKnots() + 1)]
    kv = np.array([k for k, m in zip(knots, mults) for _ in range(m)])
    kv = (kv - kv.min()) / (kv.max() - kv.min())
    if bspline.Weights() is None:
        weights = np.ones(len(poles))
    else:
        weights = np.array(list(bspline.Weights()))
    return control_points, kv, weights, degree


# ---------------------------------------------------------------------------
# Convertor classes  (ported from myclass.py)
# ---------------------------------------------------------------------------

class GeomCurve3dConvertor:
    """Converts an OCC 3-D edge curve into a :class:`~cadmap.geometry.nurbs.Curve3d`."""

    def __init__(self, edge: Topology, curv_idx: str) -> None:
        self.edge_topology = edge
        self.curv_idx = curv_idx

    def convert(self) -> Curve3d:
        adaptor = BRepAdaptor_Curve(self.edge_topology.TopoDS)
        bspline = _curve3d_adaptor_to_bspline(adaptor)
        ctp, kv, weights, degree = _extract_curve_data(bspline)
        periodic = bspline.IsPeriodic()
        if periodic:
            bspline.SetNotPeriodic()
            ctp, kv, weights, degree = _extract_curve_data(bspline)
            periodic = False
        return Curve3d(
            init_ctp=ctp,
            knotvector=kv,
            curv_id=self.curv_idx,
            periodic=periodic,
            degree=degree,
            weight=weights,
        )


class GeomCurve2dConvertor:
    """Converts an OCC 2-D pcurve into a :class:`~cadmap.geometry.nurbs.Curve2d`."""

    def __init__(self, face: Topology, edge: Topology, curve2d_idx: str) -> None:
        self.face_topology = face
        self.face = face.TopoDS
        self.edge = edge.TopoDS
        self.curve2d_idx = curve2d_idx

    def convert(self) -> Curve2d:
        from OCC.Core.Geom2d import Geom2d_BSplineCurve, Handle_Geom2d_BSplineCurve_DownCast

        reverse = (self.edge.Orientation() == TopAbs_REVERSED) ^ \
                  (self.face.Orientation() == TopAbs_REVERSED)

        original_surf = BRep_Tool.Surface(self.face)
        new_surf = self.face_topology.geom._to_geomsurface()

        curve2d_adaptor = BRepAdaptor_Curve2d(self.edge, self.face)

        if original_surf.DynamicType().Name() != "Geom_BSplineSurface":
            CurveBspline = _convert_2d_curve_by_sampling(self.face, new_surf, self.edge)
        else:
            u_min, u_max, v_min, v_max = original_surf.Bounds()
            u_shift, u_scale = -u_min, u_max - u_min
            v_shift, v_scale = -v_min, v_max - v_min
            t = curve2d_adaptor.GetType()

            if t == GeomAbs_Line:
                CurveBspline = _scale_line_2d(curve2d_adaptor, u_shift, u_scale, v_shift, v_scale)
            elif t == GeomAbs_Circle:
                CurveBspline = _circle_2d_to_bspline(curve2d_adaptor)
            elif t == GeomAbs_BSplineCurve:
                curve2d_geom, _, _ = BRep_Tool.CurveOnSurface(self.edge, self.face)
                CurveBspline = Geom2d_BSplineCurve.DownCast(
                    Handle_Geom2d_BSplineCurve_DownCast(curve2d_geom).Copy()
                )
                for i in range(1, CurveBspline.NbPoles() + 1):
                    p = CurveBspline.Pole(i)
                    CurveBspline.SetPole(
                        i,
                        gp_Pnt2d((p.X() + u_shift) / u_scale, (p.Y() + v_shift) / v_scale),
                    )
            else:
                raise NotImplementedError(f"Unsupported 2-D curve type: {t}")

        ctp, kv, weights, degree = _extract_curve2d_data(CurveBspline)
        periodic = CurveBspline.IsPeriodic()
        if periodic:
            CurveBspline.SetNotPeriodic()
            ctp, kv, weights, degree = _extract_curve2d_data(CurveBspline)
            periodic = False

        if reverse:
            ctp = ctp[::-1, :]
            weights = weights[::-1]
            kv = list(reversed([kv[0] + kv[-1] - k for k in kv]))
            kv = np.array(kv)

        return Curve2d(
            init_ctp=ctp,
            knotvector=kv,
            curv_id=self.curve2d_idx,
            periodic=periodic,
            degree=degree,
            weight=weights,
            orientation=int(self.edge.Orientation()),
        )


def _extract_curve2d_data(bspline) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    degree = bspline.Degree()
    poles = [bspline.Pole(i).Coord() for i in range(1, bspline.NbPoles() + 1)]
    ctp = np.array(poles)
    knots = [bspline.Knot(i) for i in range(1, bspline.NbKnots() + 1)]
    mults = [bspline.Multiplicity(i) for i in range(1, bspline.NbKnots() + 1)]
    kv = np.array([k for k, m in zip(knots, mults) for _ in range(m)])
    kv = (kv - kv.min()) / (kv.max() - kv.min())
    if bspline.Weights() is None:
        weights = np.ones(len(poles))
    else:
        weights = np.array(list(bspline.Weights()))
    return ctp, kv, weights, degree


def _scale_line_2d(adaptor, u_shift, u_scale, v_shift, v_scale) -> Geom2d_BSplineCurve:
    p1 = adaptor.Value(adaptor.FirstParameter())
    p2 = adaptor.Value(adaptor.LastParameter())
    p1.SetX((p1.X() + u_shift) / u_scale); p1.SetY((p1.Y() + v_shift) / v_scale)
    p2.SetX((p2.X() + u_shift) / u_scale); p2.SetY((p2.Y() + v_shift) / v_scale)
    pts = TColgp_Array1OfPnt2d(1, 2)
    pts.SetValue(1, p1); pts.SetValue(2, p2)
    knots = TColStd_Array1OfReal(1, 2); mults = TColStd_Array1OfInteger(1, 2)
    knots.SetValue(1, 0.0); knots.SetValue(2, 1.0)
    mults.SetValue(1, 2); mults.SetValue(2, 2)
    return Geom2d_BSplineCurve(pts, knots, mults, 1)


def _circle_2d_to_bspline(adaptor) -> Geom2d_BSplineCurve:
    """Approximate a 2-D circle as a periodic NURBS curve then segment it."""
    from OCC.Core.ElCLib import elclib
    Circ = adaptor.Circle()
    radius = Circ.Radius()
    center = list(Circ.Location().Coord())
    start_point = elclib.Value(0, Circ).Coord()
    start_angle = np.arctan2(start_point[1] - center[1], start_point[0] - center[0])
    bspline = _create_bspline_circle2d(radius, center, start_angle)
    bspline.Segment(adaptor.FirstParameter(), adaptor.LastParameter())
    return bspline


def _create_bspline_circle2d(radius: float, center, start_angle: float = 0.0) -> Geom2d_BSplineCurve:
    w_mid, w_corner = 1.0, 2 ** 0.5 / 2.0
    poles_data = np.array([
        [radius, 0], [radius, radius], [0, radius], [-radius, radius],
        [-radius, 0], [-radius, -radius], [0, -radius], [radius, -radius],
    ])
    if start_angle != 0.0:
        c, s = np.cos(start_angle), np.sin(start_angle)
        poles_data = (np.array([[c, -s], [s, c]]) @ poles_data.T).T
    poles_data = poles_data + np.array(center)
    poles_array = TColgp_Array1OfPnt2d(1, 8)
    for i, (x, y) in enumerate(poles_data):
        poles_array.SetValue(i + 1, gp_Pnt2d(float(x), float(y)))
    weights_array = TColStd_Array1OfReal(1, 8)
    for i, w in enumerate([w_mid, w_corner] * 4):
        weights_array.SetValue(i + 1, w)
    knots = TColStd_Array1OfReal(1, 5); mults = TColStd_Array1OfInteger(1, 5)
    for i in range(5):
        knots.SetValue(i + 1, float(i)); mults.SetValue(i + 1, 2)
    return Geom2d_BSplineCurve(poles_array, weights_array, knots, mults, 2, True)


def _convert_2d_curve_by_sampling(face, new_surf, edge, num_samples: int = 100) -> Geom2d_BSplineCurve:
    """Fallback: approximate the 2-D curve by projecting 3-D samples onto new_surf."""
    from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
    adaptor3d = BRepAdaptor_Curve(edge, face)
    params = np.linspace(adaptor3d.FirstParameter(), adaptor3d.LastParameter(), num_samples)
    pts2d = TColgp_Array1OfPnt2d(1, num_samples)
    for idx, t in enumerate(params):
        p3d = adaptor3d.Value(t)
        proj = GeomAPI_ProjectPointOnSurf(p3d, new_surf)
        u, v = proj.LowerDistanceParameters()
        pts2d.SetValue(idx + 1, gp_Pnt2d(u, v))
    from OCC.Core.Geom2dAPI import Geom2dAPI_PointsToBSpline
    from OCC.Core.TColgp import TColgp_HArray1OfPnt2d
    h_pts = TColgp_HArray1OfPnt2d(1, num_samples)
    for i in range(1, num_samples + 1):
        h_pts.SetValue(i, pts2d.Value(i))
    approx = Geom2dAPI_PointsToBSpline(h_pts, 3, 8, 3, 1e-6)
    return approx.Curve()


class GeomSurfaceConvertor:
    """Converts an OCC face surface into a :class:`~cadmap.geometry.nurbs.BsplineSurface`."""

    def __init__(self, face: Topology, surf_idx: str, ctp_start: int) -> None:
        self.face_topology = face
        self.surf_idx = surf_idx
        self.ctp_start = ctp_start

    def convert(self) -> BsplineSurface:
        from OCC.Core.Convert import (
            Convert_ConeToBSplineSurface,
            Convert_CylinderToBSplineSurface,
            Convert_SphereToBSplineSurface,
            Convert_TorusToBSplineSurface,
        )

        face = self.face_topology.TopoDS
        surf_adaptor = BRepAdaptor_Surface(face, False)
        t = surf_adaptor.GetType()

        if t == GeomAbs_BSplineSurface:
            occ_surf = surf_adaptor.BSpline()
        elif t == GeomAbs_Plane:
            occ_surf = BRepAdaptor_Surface(
                BRepBuilderAPI_NurbsConvert(face).Shape(), False
            ).BSpline()
        elif t == GeomAbs_Cylinder:
            occ_surf = geomconvert.SurfaceToBSplineSurface(
                Convert_CylinderToBSplineSurface(surf_adaptor.Cylinder()).BSplineSurface()
            )
        elif t == GeomAbs_Cone:
            occ_surf = geomconvert.SurfaceToBSplineSurface(
                Convert_ConeToBSplineSurface(surf_adaptor.Cone()).BSplineSurface()
            )
        elif t == GeomAbs_Sphere:
            occ_surf = geomconvert.SurfaceToBSplineSurface(
                Convert_SphereToBSplineSurface(surf_adaptor.Sphere()).BSplineSurface()
            )
        elif t == GeomAbs_Torus:
            occ_surf = geomconvert.SurfaceToBSplineSurface(
                Convert_TorusToBSplineSurface(surf_adaptor.Torus()).BSplineSurface()
            )
        else:
            raise NotImplementedError(f"Unsupported surface type: {t}")

        return self._occ_surf_to_bsplinesurface(occ_surf)

    def _occ_surf_to_bsplinesurface(self, occ_surf) -> BsplineSurface:
        u_min, u_max, v_min, v_max = occ_surf.Bounds()
        su, sv = occ_surf.NbUPoles(), occ_surf.NbVPoles()

        ctp = np.array([
            [occ_surf.Pole(i + 1, j + 1).Coord() for j in range(sv)]
            for i in range(su)
        ])
        weights = np.array([
            [occ_surf.Weight(i + 1, j + 1) for j in range(sv)]
            for i in range(su)
        ])

        knots_u = [occ_surf.UKnot(i) for i in range(1, occ_surf.NbUKnots() + 1)]
        mults_u = [occ_surf.UMultiplicity(i) for i in range(1, occ_surf.NbUKnots() + 1)]
        kv_u = np.array([k for k, m in zip(knots_u, mults_u) for _ in range(m)])
        kv_u = (kv_u - u_min) / (u_max - u_min)

        knots_v = [occ_surf.VKnot(i) for i in range(1, occ_surf.NbVKnots() + 1)]
        mults_v = [occ_surf.VMultiplicity(i) for i in range(1, occ_surf.NbVKnots() + 1)]
        kv_v = np.array([k for k, m in zip(knots_v, mults_v) for _ in range(m)])
        kv_v = (kv_v - v_min) / (v_max - v_min)

        periodic_u = occ_surf.IsUPeriodic()
        periodic_v = occ_surf.IsVPeriodic()

        surf = BsplineSurface(
            init_ctp=ctp,
            knotvector_u=kv_u,
            knotvector_v=kv_v,
            surface_id=self.surf_idx,
            ctp_start_index=self.ctp_start,
            periodic_u=periodic_u,
            periodic_v=periodic_v,
            size_u=su,
            size_v=sv,
            weight=weights,
        )
        surf.topology = self.face_topology
        return surf


# ---------------------------------------------------------------------------
# Topology convertor hierarchy
# ---------------------------------------------------------------------------

class VertexConvertor:
    def __init__(self, vertex: Topology) -> None:
        self.topology = vertex

    def convert(self) -> None:
        pass  # Vertices carry no geometry beyond their topological identity


class EdgeConvertor:
    def __init__(self, edge: Topology) -> None:
        self.topology = edge

    def convert(self) -> None:
        vertex_it = TopExp_Explorer(self.topology.TopoDS, TopAbs_VERTEX)
        while vertex_it.More():
            v = Topology(topods.Vertex(vertex_it.Current()))
            v.add_ancestor(self.topology)
            self.topology.add_descendent(v)
            VertexConvertor(v).convert()
            vertex_it.Next()


class WireConvertor:
    def __init__(self, wire: Topology) -> None:
        self.topology = wire

    def convert(self) -> None:
        edge_it = TopExp_Explorer(self.topology.TopoDS, TopAbs_EDGE)
        while edge_it.More():
            e = Topology(topods.Edge(edge_it.Current()))
            e.add_ancestor(self.topology)
            self.topology.add_descendent(e)
            EdgeConvertor(e).convert()
            edge_it.Next()


class FaceConvertor:
    def __init__(self, face: Topology) -> None:
        self.topology = face

    def convert(self) -> None:
        wire_it = TopExp_Explorer(self.topology.TopoDS, TopAbs_WIRE)
        while wire_it.More():
            w = Topology(topods.Wire(wire_it.Current()))
            w.add_ancestor(self.topology)
            self.topology.add_descendent(w)
            WireConvertor(w).convert()
            if w.TopoDS.Orientation() == TopAbs_REVERSED:
                w.descendent.reverse()
            wire_it.Next()


class SolidConvertor:
    """Converts an OCC solid into a :class:`~cadmap.geometry.topology.Topology` tree."""

    def __init__(self, solid: Topology) -> None:
        self.topology = solid

    def convert(self) -> None:
        face_it = TopExp_Explorer(self.topology.TopoDS, TopAbs_FACE)
        while face_it.More():
            f = Topology(topods.Face(face_it.Current()))
            f.add_ancestor(self.topology)
            self.topology.add_descendent(f)
            FaceConvertor(f).convert()
            face_it.Next()


# ---------------------------------------------------------------------------
# Post-conversion helpers  (ported from myclass.py / CADMap_for_claude.py)
# ---------------------------------------------------------------------------

def find_all_targets_at_depth(solid: Topology, target_edge: Topology, depth: int) -> None:
    """Assign shared 3-D edge geometry across faces that share the same topological edge."""
    # Placeholder: implemented in the full conversion pipeline.
    pass


def add_geom_to_solid(solid: Topology) -> None:
    """Populate ``.geom`` and ``.geom2d`` on every node in the topology tree."""
    surf_counter = 0
    curve3d_counter = 0
    curve2d_counter = 0
    ctp_offset = 0

    all_curve3d: dict = {}  # OCC edge hash → Curve3d

    for face_topo in solid.descendent:
        # --- surface ---
        surf_conv = GeomSurfaceConvertor(face_topo, f"Surface_{surf_counter}", ctp_offset)
        surf = surf_conv.convert()
        face_topo.geom = surf
        ctp_offset += surf.size_u * surf.size_v
        surf_counter += 1

        for wire_topo in face_topo.descendent:
            for edge_topo in wire_topo.descendent:
                # --- 3-D curve (shared across faces) ---
                edge_key = hash(edge_topo.TopoDS)
                if edge_key not in all_curve3d:
                    c3d_conv = GeomCurve3dConvertor(edge_topo, f"Curve3d_{curve3d_counter}")
                    c3d = c3d_conv.convert()
                    all_curve3d[edge_key] = c3d
                    curve3d_counter += 1
                edge_topo.geom = all_curve3d[edge_key]

                # --- 2-D pcurve ---
                c2d_conv = GeomCurve2dConvertor(face_topo, edge_topo, f"Curve2d_{curve2d_counter}")
                c2d = c2d_conv.convert()
                c2d.topology = edge_topo
                edge_topo.geom2d = c2d
                curve2d_counter += 1


def solid_to_device(solid: Topology, device: str = "cpu") -> None:
    """Move all surface ``DiffSurf`` modules to *device*."""
    for face in solid.descendent:
        if face.geom is not None:
            face.geom.to_device(device)


def create_diffsurf_of_solid(solid: Topology) -> None:
    """Instantiate NURBSDiff evaluators for every face in *solid*."""
    for face in solid.descendent:
        if face.geom is not None:
            face.geom.create_diff_NURBS()


def compute_inside_evalpts(solid: Topology) -> None:
    """Compute and store the trim-region mask (``inside_evalpts``) for every face.

    Must be called after :func:`add_geom_to_solid` and before
    :class:`~cadmap.deformation.fitting.CADMapOptimizer`.  Sets
    ``face.geom.inside_evalpts`` to a boolean ``(uEvalPtSize, vEvalPtSize)``
    array that is ``True`` where the evaluation grid point lies inside the
    parametric trim boundary.
    """
    import numpy as np
    from cadmap._internal.math_ops import batch_wn

    for face in solid.descendent:
        surf = face.geom
        if surf is None:
            continue
        u_min, u_max, v_min, v_max = surf.get_UVBounds()
        u, v = np.meshgrid(
            np.linspace(u_min, u_max, surf.uEvalPtSize),
            np.linspace(v_min, v_max, surf.vEvalPtSize),
            indexing="ij",
        )
        points = np.stack([u, v], axis=-1)
        surf.inside_evalpts = batch_wn(points, face)
