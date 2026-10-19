"""OCC topology-to-geometry conversion: reads a B-Rep solid and populates
the Topology tree with BsplineSurface / Curve3d / Curve2d geometry objects.

The public entry point is :func:`step_to_solid`, which wraps the topology walk
(:class:`SolidConvertor`) and the geometry conversion (:func:`add_geom_to_solid`):

    solid = step_to_solid(path)
    solid_to_device(solid, "cpu")
    create_diffsurf_of_solid(solid)

The inverse direction -- rebuilding an OCC solid and writing it back to STEP --
lives in :mod:`cadmap.conversion.export`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert
from OCC.Core.Geom import Geom_BSplineCurve, Geom_BSplineSurface
from OCC.Core.Geom2d import Geom2d_BSplineCurve
from OCC.Core.GeomAbs import (
    GeomAbs_BSplineSurface,
    GeomAbs_BezierSurface,
    GeomAbs_C2,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_OffsetSurface,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_SurfaceOfExtrusion,
    GeomAbs_SurfaceOfRevolution,
    GeomAbs_Torus,
)
from OCC.Core.gp import gp_Pnt2d
from OCC.Core.TColgp import TColgp_Array1OfPnt2d
from OCC.Core.TopAbs import (
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_REVERSED,
    TopAbs_VERTEX,
    TopAbs_WIRE,
)
from OCC.Core.TopoDS import topods
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

#: Tolerance and limits used when a curve has to be approximated rather than
#: converted exactly (offset / intersection curves on fillets and blends).
_CURVE_APPROX_TOL = 1e-7

#: Parameter ranges shorter than this are treated as no pcurve at all.
_PCURVE_MIN_RANGE = 1e-12
_CURVE_APPROX_MAX_SEGMENTS = 200
_CURVE_APPROX_MAX_DEGREE = 8

#: Surface types that ``BRepBuilderAPI_NurbsConvert`` turns into a B-spline
#: while keeping the face's own parametric bounds and pcurves.
_NURBS_CONVERTIBLE_SURFACES = frozenset({
    GeomAbs_Plane,
    GeomAbs_Cylinder,
    GeomAbs_Cone,
    GeomAbs_Sphere,
    GeomAbs_Torus,
    GeomAbs_BezierSurface,
    GeomAbs_C2,
    GeomAbs_SurfaceOfRevolution,
    GeomAbs_SurfaceOfExtrusion,
    GeomAbs_OffsetSurface,
})


def _to_non_periodic_surface(occ_surf: Geom_BSplineSurface) -> Geom_BSplineSurface:
    """Return a clamped, non-periodic copy of *occ_surf*.

    Closed surfaces (cylinders, spheres, tori) convert to periodic B-splines
    whose pole count does not match ``len(knotvector) - degree - 1``, which the
    :class:`~cadmap.geometry.nurbs.BsplineSurface` representation and NURBSDiff
    both assume.  Unwrapping to the non-periodic form duplicates the wrap-around
    poles and leaves the surface geometrically identical.
    """
    if not (occ_surf.IsUPeriodic() or occ_surf.IsVPeriodic()):
        return occ_surf

    surf = Geom_BSplineSurface.DownCast(occ_surf.Copy())
    if surf.IsUPeriodic():
        surf.SetUNotPeriodic()
    if surf.IsVPeriodic():
        surf.SetVNotPeriodic()
    return surf


def _edge_to_bspline3d(edge) -> Optional[Geom_BSplineCurve]:
    """Convert the 3-D curve of *edge*, restricted to the edge, to a B-spline.

    Conics and Beziers convert exactly; anything OpenCascade cannot convert
    exactly (offset and intersection curves, which show up on fillets) is
    approximated to ``_CURVE_APPROX_TOL``.  Returns ``None`` for degenerate
    edges, which carry no 3-D curve at all (the poles of a sphere, the apex of
    a cone).
    """
    from OCC.Core.Geom import Geom_TrimmedCurve
    from OCC.Core.GeomConvert import GeomConvert_ApproxCurve, geomconvert

    if BRep_Tool.Degenerated(edge):
        return None

    curve, first, last = BRep_Tool.Curve(edge)
    if curve is None:
        return None

    try:
        return geomconvert.CurveToBSplineCurve(Geom_TrimmedCurve(curve, first, last))
    except RuntimeError:
        approx = GeomConvert_ApproxCurve(
            Geom_TrimmedCurve(curve, first, last),
            _CURVE_APPROX_TOL, GeomAbs_C2, _CURVE_APPROX_MAX_SEGMENTS, _CURVE_APPROX_MAX_DEGREE,
        )
        if not approx.IsDone():
            raise NotImplementedError(
                f"Could not convert the 3-D curve of edge {edge} to a B-spline."
            )
        return approx.Curve()


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

    def convert(self) -> Optional[Curve3d]:
        """Return the edge's 3-D curve, or ``None`` if the edge is degenerate."""
        bspline = _edge_to_bspline3d(self.edge_topology.TopoDS)
        if bspline is None:
            return None
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


def _pcurve_to_bspline2d(edge, face) -> Optional[Geom2d_BSplineCurve]:
    """Convert the pcurve of *edge* on *face* to a B-spline, in the face's own
    parameter space.  Returns ``None`` when the edge carries no pcurve."""
    from OCC.Core.Geom2d import Geom2d_TrimmedCurve
    from OCC.Core.Geom2dConvert import Geom2dConvert_ApproxCurve, geom2dconvert

    curve2d, first, last = BRep_Tool.CurveOnSurface(edge, face)
    if curve2d is None or last - first < _PCURVE_MIN_RANGE:
        return None

    trimmed = Geom2d_TrimmedCurve(curve2d, first, last)
    try:
        bspline = geom2dconvert.CurveToBSplineCurve(trimmed)
    except RuntimeError:
        approx = Geom2dConvert_ApproxCurve(
            trimmed, _CURVE_APPROX_TOL, GeomAbs_C2,
            _CURVE_APPROX_MAX_SEGMENTS, _CURVE_APPROX_MAX_DEGREE,
        )
        if not approx.IsDone():
            return None
        bspline = approx.Curve()

    if bspline.IsPeriodic():
        bspline.SetNotPeriodic()
    return bspline


def _rescale_pcurve(
    bspline: Geom2d_BSplineCurve,
    u_min: float, u_max: float, v_min: float, v_max: float,
) -> None:
    """Map a pcurve's poles from ``[u_min, u_max] x [v_min, v_max]`` into the
    unit square in place, matching the knot normalisation applied to surfaces."""
    u_scale = u_max - u_min
    v_scale = v_max - v_min
    for i in range(1, bspline.NbPoles() + 1):
        p = bspline.Pole(i)
        bspline.SetPole(
            i, gp_Pnt2d((p.X() - u_min) / u_scale, (p.Y() - v_min) / v_scale)
        )


class GeomCurve2dConvertor:
    """Converts an OCC 2-D pcurve into a :class:`~cadmap.geometry.nurbs.Curve2d`."""

    def __init__(
        self,
        face: Topology,
        edge: Topology,
        curve2d_idx: str,
        pcurve_face=None,
        pcurve_edge=None,
    ) -> None:
        self.face_topology = face
        self.face = face.TopoDS
        self.edge = edge.TopoDS
        self.curve2d_idx = curve2d_idx
        # Where to read the pcurve from: the NURBS-converted face and edge when
        # the surface was converted, otherwise the original pair.
        self.pcurve_face = pcurve_face if pcurve_face is not None else self.face
        self.pcurve_edge = pcurve_edge if pcurve_edge is not None else self.edge

    def convert(self) -> Curve2d:
        reverse = (
            (self.edge.Orientation() == TopAbs_REVERSED)
            ^ (self.face.Orientation() == TopAbs_REVERSED)
        )

        surf = self.face_topology.geom
        CurveBspline = _pcurve_to_bspline2d(self.pcurve_edge, self.pcurve_face)

        if CurveBspline is not None:
            # The pcurve lives in the parameter space of the original face;
            # rescale it into the [0, 1]^2 space the converted surface uses.
            _rescale_pcurve(CurveBspline, *surf.param_bounds)
        else:
            # No usable pcurve: recover it by projecting samples of the 3-D
            # curve onto the converted surface, which is already normalised.
            CurveBspline = _convert_2d_curve_by_sampling(
                self.face, surf._to_geomsurface(), self.edge
            )

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
        # Set by convert() when the surface goes through NurbsConvert; the
        # converted face carries the pcurves that match the converted surface.
        self.nurbs_convert: Optional[BRepBuilderAPI_NurbsConvert] = None
        self.nurbs_face = None

    def convert(self) -> BsplineSurface:
        face = self.face_topology.TopoDS
        surf_adaptor = BRepAdaptor_Surface(face, False)
        t = surf_adaptor.GetType()

        if t == GeomAbs_BSplineSurface:
            occ_surf = surf_adaptor.BSpline()
        elif t in _NURBS_CONVERTIBLE_SURFACES:
            # Planes, cylinders, cones, spheres, tori, surfaces of revolution
            # and extrusion all convert through the same B-Rep NURBS
            # conversion, which -- unlike the analytic
            # Convert_*ToBSplineSurface helpers -- keeps the face's own
            # parametric bounds.  It does NOT keep the parameterisation: a
            # rational arc is not parameterised by the cylinder's angle, so
            # the converted face's own pcurves (``nurbs_face``) are the only
            # correct ones.  Rescaling the original face's pcurves would skew
            # every trim curve that is not an isoparametric line.
            self.nurbs_convert = BRepBuilderAPI_NurbsConvert(face, True)
            self.nurbs_face = topods.Face(self.nurbs_convert.Shape())
            occ_surf = BRepAdaptor_Surface(self.nurbs_face, False).BSpline()
        else:
            raise NotImplementedError(f"Unsupported surface type: {t}")

        occ_surf = _to_non_periodic_surface(occ_surf)
        return self._occ_surf_to_bsplinesurface(occ_surf)

    def converted_edge(self, edge):
        """Return *edge* as it appears on the NURBS-converted face.

        Falls back to *edge* itself when the surface needed no conversion, or
        when the conversion did not modify this particular edge.  The original
        orientation is re-applied: ``ModifiedShape`` drops it, and a seam edge
        needs it to select which of its two pcurves is returned.
        """
        if self.nurbs_convert is None:
            return edge
        modified = self.nurbs_convert.ModifiedShape(edge)
        if modified.IsNull():
            return edge
        converted = topods.Edge(modified)
        converted.Orientation(edge.Orientation())
        return converted

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
            param_bounds=(u_min, u_max, v_min, v_max),
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
                # Degenerate edges (sphere poles, cone apices) have no 3-D
                # curve; ``convert`` returns None and ``.geom`` stays unset.
                edge_key = hash(edge_topo.TopoDS)
                if edge_key not in all_curve3d:
                    c3d = GeomCurve3dConvertor(edge_topo, f"Curve3d_{curve3d_counter}").convert()
                    all_curve3d[edge_key] = c3d
                    if c3d is not None:
                        curve3d_counter += 1
                edge_topo.geom = all_curve3d[edge_key]

                # --- 2-D pcurve (read off the converted face, see
                #     GeomSurfaceConvertor.convert) ---
                c2d_conv = GeomCurve2dConvertor(
                    face_topo,
                    edge_topo,
                    f"Curve2d_{curve2d_counter}",
                    pcurve_face=surf_conv.nurbs_face,
                    pcurve_edge=surf_conv.converted_edge(edge_topo.TopoDS),
                )
                c2d = c2d_conv.convert()
                c2d.topology = edge_topo
                edge_topo.geom2d = c2d
                curve2d_counter += 1


def step_to_solid(path: str, add_geom: bool = True) -> Topology:
    """Read a STEP file and return a fully populated cadmap topology tree.

    Convenience wrapper around ``read_step_file`` → :class:`Topology` →
    :class:`SolidConvertor` → :func:`add_geom_to_solid`.  The returned root
    node's ``descendent`` list holds one face node per B-Rep face, each
    carrying a :class:`~cadmap.geometry.nurbs.BsplineSurface` in ``.geom``;
    its edges carry a :class:`~cadmap.geometry.nurbs.Curve3d` in ``.geom`` and
    a :class:`~cadmap.geometry.nurbs.Curve2d` trim curve in ``.geom2d``.

    Set *add_geom* to ``False`` to build the topology tree only.
    """
    solid = Topology(read_step_file(str(path)))
    SolidConvertor(solid).convert()
    if add_geom:
        add_geom_to_solid(solid)
    return solid


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
