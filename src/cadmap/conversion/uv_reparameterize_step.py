"""Geometry-preserving automatic UV reparameterization of a STEP solid.

Only face charts change: every face keeps its point set, orientation, edges and
vertices. Two exact chart changes are applied, each only where it pays off:

* ``restrict``: a single-span B-spline face whose trim covers a small part of
  its parameter domain is segmented to a padded, conservative trim enclosure.
  The parameterization is unchanged, so the pcurves are kept as they are.
* ``strip``: a plane or single-span B-spline face whose trim still occupies
  less than the trigger fraction of its enclosure gets the buffered polynomial
  strip chart of :mod:`cadmap.geometry.trim_uv_diagnostics`. The new surface is
  the exact homogeneous substitution, and every pcurve is the exact inverse.

Every changed face is then checked on the final OCC objects, independently of
the algebra that built it: surface position and normal orientation against the
original face, regularity, pcurves inside the new domain, and pcurve-on-surface
versus the 3D edge curve. A face failing any check keeps its original chart.
The whole shape must afterwards be no less valid than the input, with equal
topology counts, volume and area, and the same must hold after STEP readback.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np

from cadmap.geometry.composition import extract_bezier
from cadmap.geometry.trim_uv_diagnostics import TrimUVOptions, curve_bounds, select_automatic_maps

SCHEMA = 'cadmap-uv-reparameterization/1'


@dataclass(frozen=True)
class StepUVOptions:
    trim: TrimUVOptions = field(default_factory=TrimUVOptions)
    restrict_bspline: bool = True
    padding_fraction: float = .002      # of the trim enclosure, per side
    geometry_tolerance: float = 1e-7    # mm, sampled new-vs-original surface
    edge_tolerance: float = 1e-6        # mm, allowed pcurve/3D-curve mismatch
    volume_rtol: float = 1e-7
    grid: int = 17
    curve_samples: int = 65

    def __post_init__(self):
        if (not 0 < self.padding_fraction < .5 or min(self.geometry_tolerance, self.edge_tolerance,
                                                      self.volume_rtol) <= 0
                or self.grid < 3 or self.curve_samples < 3):
            raise ValueError('invalid STEP UV reparameterization options')


# ---------------------------------------------------------------------------
# OCC helpers
# ---------------------------------------------------------------------------

def _indexed(shape, kind):
    from OCC.Core.TopExp import topexp
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape
    m = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, kind, m)
    return [m.FindKey(i) for i in range(1, m.Size()+1)]


def _faces(shape):
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopoDS import topods
    return [topods.Face(f) for f in _indexed(shape, TopAbs_FACE)]


def _occurrences(face):
    """Edges of each wire, in a fixed exploration order shared by copies."""
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_WIRE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods
    rows, wires = [], TopExp_Explorer(face, TopAbs_WIRE)
    wire = 0
    while wires.More():
        edges = TopExp_Explorer(wires.Current(), TopAbs_EDGE)
        while edges.More():
            rows.append((wire, topods.Edge(edges.Current())))
            edges.Next()
        wire += 1
        wires.Next()
    return rows


def counts(shape):
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID, TopAbs_VERTEX
    return {name: len(_indexed(shape, kind)) for name, kind in
            [('solids', TopAbs_SOLID), ('shells', TopAbs_SHELL), ('faces', TopAbs_FACE),
             ('edges', TopAbs_EDGE), ('vertices', TopAbs_VERTEX)]}


def _mass(shape):
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps
    v, a = GProp_GProps(), GProp_GProps()
    brepgprop.VolumeProperties(shape, v)
    brepgprop.SurfaceProperties(shape, a)
    return float(v.Mass()), float(a.Mass())


def _face_areas(shape):
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps
    areas = []
    for f in _faces(shape):
        p = GProp_GProps()
        brepgprop.SurfaceProperties(f, p)
        areas.append(float(p.Mass()))
    return np.sort(areas)


def shape_status(shape):
    from cadmap.deformation.validity import brep_status
    volume, area = _mass(shape)
    return {**asdict(brep_status(shape)), 'counts': counts(shape), 'volume': volume, 'area': area}


def _pcurve_bspline(edge, face):
    """Pcurve as a nonperiodic Geom2d B-spline, and whether ``t`` is unchanged."""
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.Geom2d import Geom2d_BSplineCurve, Geom2d_TrimmedCurve
    from OCC.Core.Geom2dConvert import geom2dconvert
    curve, first, last = BRep_Tool.CurveOnSurface(edge, face)
    if curve is None:
        raise ValueError('edge has no pcurve on face')
    bspline = geom2dconvert.CurveToBSplineCurve(Geom2d_TrimmedCurve(curve, first, last))
    bspline = Geom2d_BSplineCurve.DownCast(bspline)
    if bspline.IsPeriodic():
        bspline.SetNotPeriodic()
    t = np.linspace(first, last, 9)
    same = (np.isclose(bspline.FirstParameter(), first, rtol=0, atol=1e-12)
            and np.isclose(bspline.LastParameter(), last, rtol=0, atol=1e-12)
            and max(np.hypot(*np.subtract(bspline.Value(float(x)).Coord(), curve.Value(float(x)).Coord()))
                    for x in t) <= 1e-12*max(1., np.abs(t).max()))
    return bspline, float(first), float(last), bool(same)


def _curve_source(bspline, box, metadata, reverse=False):
    """Exact Bezier pieces of a 2D B-spline in box-normalized coordinates."""
    n = bspline.NbPoles()
    poles = np.array([bspline.Pole(i).Coord() for i in range(1, n+1)])
    weights = np.array([bspline.Weight(i) for i in range(1, n+1)])
    knots = np.repeat([bspline.Knot(i) for i in range(1, bspline.NbKnots()+1)],
                      [bspline.Multiplicity(i) for i in range(1, bspline.NbKnots()+1)])
    poles = (poles - box[:, 0]) / np.diff(box, axis=1)[:, 0]
    if reverse:
        poles, weights, knots = poles[::-1], weights[::-1], (knots[0]+knots[-1]-knots)[::-1]
    return extract_bezier(poles, weights, [knots], [bspline.Degree()], metadata)


def _normalized_bspline2d(bspline, box):
    from OCC.Core.Geom2d import Geom2d_BSplineCurve
    from OCC.Core.gp import gp_Pnt2d
    copy = Geom2d_BSplineCurve.DownCast(bspline.Copy())
    for i in range(1, copy.NbPoles()+1):
        x, y = copy.Pole(i).Coord()
        copy.SetPole(i, gp_Pnt2d((x-box[0, 0])/(box[0, 1]-box[0, 0]), (y-box[1, 0])/(box[1, 1]-box[1, 0])))
    return copy


def _surface_source(surface, box):
    """Exact single Bezier patch of the raw surface over ``box`` in unit UV."""
    from OCC.Core.Geom import Geom_BSplineSurface
    kind = surface.DynamicType().Name()
    if kind == 'Geom_Plane':
        cp = np.array([[surface.Value(float(u), float(v)).Coord() for v in box[1]] for u in box[0]])
        return extract_bezier(cp, np.ones((2, 2)), [[0, 0, 1, 1]]*2, [1, 1]), 'plane'
    if kind != 'Geom_BSplineSurface':
        raise ValueError('unsupported surface type %s: only planes and B-spline surfaces' % kind)
    bs = Geom_BSplineSurface.DownCast(surface)
    if bs.IsUPeriodic() or bs.IsVPeriodic():
        raise ValueError('unsupported surface: periodic B-spline')
    segment = _segment(bs, box)
    if segment.NbUKnots() != 2 or segment.NbVKnots() != 2:
        raise ValueError('unsupported surface: multiple spans inside the trim enclosure')
    su, sv = segment.NbUPoles(), segment.NbVPoles()
    cp = np.array([[segment.Pole(i, j).Coord() for j in range(1, sv+1)] for i in range(1, su+1)])
    w = np.array([[segment.Weight(i, j) for j in range(1, sv+1)] for i in range(1, su+1)])
    du, dv = segment.UDegree(), segment.VDegree()
    kv = [np.r_[np.zeros(du+1), np.ones(du+1)], np.r_[np.zeros(dv+1), np.ones(dv+1)]]
    return extract_bezier(cp, w, kv, [du, dv]), 'bspline'


def _segment(bs, box):
    from OCC.Core.Geom import Geom_BSplineSurface
    segment = Geom_BSplineSurface.DownCast(bs.Copy())
    segment.Segment(*map(float, box.ravel()))
    return segment


def _bezier_surface(piece):
    """OCC B-spline surface of one rational Bezier patch on [0, 1]^2."""
    from OCC.Core.Geom import Geom_BSplineSurface
    from OCC.Core.TColgp import TColgp_Array2OfPnt
    from OCC.Core.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal, TColStd_Array2OfReal
    from OCC.Core.gp import gp_Pnt
    h = piece.homogeneous
    du, dv = piece.degrees
    if max(du, dv) > 25:
        raise ValueError('reparameterized surface exceeds the OCC degree limit')
    w = h[..., -1] / h[..., -1].max()
    xyz = h[..., :3] / h[..., -1:]
    poles, weights = TColgp_Array2OfPnt(1, du+1, 1, dv+1), TColStd_Array2OfReal(1, du+1, 1, dv+1)
    for i in range(du+1):
        for j in range(dv+1):
            poles.SetValue(i+1, j+1, gp_Pnt(*map(float, xyz[i, j])))
            weights.SetValue(i+1, j+1, float(w[i, j]))
    knots, mu, mv = TColStd_Array1OfReal(1, 2), TColStd_Array1OfInteger(1, 2), TColStd_Array1OfInteger(1, 2)
    knots.SetValue(1, 0.)
    knots.SetValue(2, 1.)
    for m, d in ((mu, du), (mv, dv)):
        m.SetValue(1, d+1)
        m.SetValue(2, d+1)
    if np.ptp(w) <= 64*np.finfo(float).eps:
        return Geom_BSplineSurface(poles, knots, knots, mu, mv, du, dv)
    return Geom_BSplineSurface(poles, weights, knots, knots, mu, mv, du, dv)


# ---------------------------------------------------------------------------
# Planning: choose a chart for every face, without touching the shape
# ---------------------------------------------------------------------------

def _plan_face(index, face, options):
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.BRepTools import breptools
    from OCC.Core.GeomAbs import GeomAbs_BSplineSurface, GeomAbs_Plane
    from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED
    from OCC.Core.TopLoc import TopLoc_Location
    row = {'face': index, 'action': 'unchanged'}
    kind = BRepAdaptor_Surface(face, False).GetType()
    row['surface_type'] = {GeomAbs_Plane: 'plane', GeomAbs_BSplineSurface: 'bspline'}.get(kind, int(kind))
    if kind not in (GeomAbs_Plane, GeomAbs_BSplineSurface):
        row['reason'] = 'surface type kept as is'
        return row, None
    forward = face.Oriented(TopAbs_FORWARD)
    occurrences = _occurrences(forward)
    if not occurrences:
        row['reason'] = 'face has no trim'
        return row, None
    curves, natives = [], []
    for k, (wire, edge) in enumerate(occurrences):
        if BRep_Tool.Degenerated(edge) or BRep_Tool.IsClosed(edge, face):
            row['reason'] = 'unsupported: degenerate or seam edge'
            return row, None
        natives.append(_pcurve_bspline(edge, face))
    # Conservative trim enclosure in native UV (Bernstein hulls, not samples).
    unit = np.array([[0., 1.], [0., 1.]])
    native_curves = [_curve_source(b, unit, {}) for b, *_ in natives]
    lo, hi, _ = curve_bounds(native_curves, options.trim.bound_tolerance)
    pad = options.padding_fraction*(hi-lo)
    box = np.column_stack((lo-pad, hi+pad))
    loc = TopLoc_Location()
    surface = BRep_Tool.Surface(face, loc)
    if row['surface_type'] == 'bspline':
        u0, u1, v0, v1 = surface.Bounds()
        domain = np.array([[u0, u1], [v0, v1]])
        box = np.column_stack((np.maximum(box[:, 0], domain[:, 0]), np.minimum(box[:, 1], domain[:, 1])))
    else:
        u0, u1, v0, v1 = breptools.UVBounds(face)
        domain = np.array([[u0, u1], [v0, v1]])
    if np.any(np.diff(box, axis=1) <= 0):
        row['reason'] = 'degenerate trim enclosure'
        return row, None
    sources = {}
    for k, ((wire, edge), (b, first, last, same)) in enumerate(zip(occurrences, natives)):
        sources[f'pcurve_{k}'] = _curve_source(
            b, box, {'face': index, 'wire': wire, 'pcurve': f'pcurve_{k}'},
            reverse=edge.Orientation() == TopAbs_REVERSED)
    try:
        surface_source, _ = _surface_source(surface, box)
    except ValueError as error:
        row['reason'] = str(error)
        return row, None
    sources[f'face_{index}'] = surface_source
    maps, report, exact, _ = select_automatic_maps(sources, options.trim, field_degree_sum=1)
    selection = report['faces'][0]
    box_area = float(np.prod(np.diff(box, axis=1)))
    domain_gain = float(np.prod(np.diff(domain, axis=1)))/box_area
    row.update({'trim_enclosure_native_uv': box.tolist(), 'native_domain_uv': domain.tolist(),
                'occupancy_in_enclosure': selection.get('original_occupancy'),
                'occupancy_in_native_domain': (selection.get('original_occupancy') or 0.)/domain_gain,
                'pcurves': len(occurrences),
                'pcurves_parameter_preserving': int(sum(n[3] for n in natives)),
                'selection_reason': selection.get('reason')})
    plan = {'box': box, 'occurrences': occurrences, 'natives': natives, 'surface': surface}
    if f'face_{index}' in maps:
        mapping = maps[f'face_{index}']
        plan.update(action='strip', mapping=mapping, new_surface=_bezier_surface(exact[f'face_{index}'].pieces[0]))
        candidate = selection['selected_candidate']
        row.update({'action': 'strip', 'map': mapping.metadata(), 'occupancy_after': candidate['occupancy'],
                    'coverage_margin_new_uv': candidate['coverage']['minimum_margin_new_uv'],
                    'source_invariance': candidate['source_invariance'],
                    'support_regularity': candidate['support_regularity']})
    elif (row['surface_type'] == 'bspline' and options.restrict_bspline
          and domain_gain >= options.trim.minimum_area_gain):
        plan.update(action='restrict', new_surface=_segment(surface, box))
        row.update({'action': 'restrict', 'occupancy_after': row['occupancy_in_enclosure'],
                    'area_gain': domain_gain})
    else:
        row['reason'] = selection.get('reason', 'no beneficial chart')
        return row, None
    row['new_degrees'] = [plan['new_surface'].UDegree(), plan['new_surface'].VDegree()]
    return row, plan


def _new_pcurves(plan):
    """New pcurve per occurrence and whether it is same-parameter by construction."""
    if plan['action'] == 'restrict':
        return [(None, same) for _, _, _, same in plan['natives']]
    out = []
    for b, first, last, same in plan['natives']:
        curve, _ = plan['mapping'].transform_pcurve(_normalized_bspline2d(b, plan['box']))
        out.append((curve, same))
    return out


# ---------------------------------------------------------------------------
# Application and independent checks
# ---------------------------------------------------------------------------

def _apply(face, plan, pcurves):
    """Swap the face surface and its pcurves in place; topology is shared."""
    from OCC.Core.BRep import BRep_Builder, BRep_Tool
    from OCC.Core.BRepLib import breplib
    from OCC.Core.Geom import Geom_BSplineSurface
    from OCC.Core.Geom2d import Geom2d_Curve
    from OCC.Core.TopLoc import TopLoc_Location
    builder = BRep_Builder()
    loc = TopLoc_Location()
    old_surface = BRep_Tool.Surface(face, loc)
    new_surface = Geom_BSplineSurface.DownCast(plan['new_surface'].Copy())
    occurrences = _occurrences(face)
    if len(occurrences) != len(plan['occurrences']):
        raise RuntimeError('face copy changed its edge occurrences')
    saved = []
    for (_, edge), (b, first, last, _), (curve, same) in zip(occurrences, plan['natives'], pcurves):
        native, f, l = BRep_Tool.CurveOnSurface(edge, face)
        saved.append((edge, curve if curve is not None else native, first, last, same))
    builder.UpdateFace(face, new_surface, loc, BRep_Tool.Tolerance(face))
    reparam = []
    for edge, curve, first, last, same in saved:
        tolerance = BRep_Tool.Tolerance(edge)
        builder.UpdateEdge(edge, curve, new_surface, loc, tolerance)
        builder.Range(edge, new_surface, loc, first, last)
        builder.UpdateEdge(edge, Geom2d_Curve.DownCast(None), old_surface, loc, tolerance)
        if not same:
            reparam.append(edge)
    for edge in reparam:
        builder.SameParameter(edge, False)
        breplib.SameParameter(edge, 1e-7)
    return occurrences


def _check_face(original, new, plan, options):
    """Compare final OCC geometry of the new face with the original face."""
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
    from OCC.Core.gp import gp_Pnt, gp_Vec
    old_s, new_s = BRepAdaptor_Surface(original, False), BRepAdaptor_Surface(new, False)

    def d1(adaptor, uv):
        out = []
        for u, v in uv:
            p, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
            adaptor.D1(float(u), float(v), p, du, dv)
            out.append((p.Coord(), np.cross(du.Coord(), dv.Coord())))
        return np.array([o[0] for o in out]), np.array([o[1] for o in out])

    box = plan['box']
    if plan['action'] == 'strip':
        to_native = lambda uv: plan['mapping'].to_original(uv)*np.diff(box, axis=1)[:, 0]+box[:, 0]
        g = np.linspace(0, 1, options.grid)
    else:
        to_native = lambda uv: uv
        g = None
    # Surface: interior grid of the whole new support plus every pcurve sample.
    uv_new, uv_old = [], []
    if g is not None:
        grid = np.stack(np.meshgrid(g, g, indexing='ij'), -1).reshape(-1, 2)
        uv_new.append(grid)
        uv_old.append(to_native(grid))
    else:
        grid = np.stack(np.meshgrid(*[np.linspace(*b, options.grid) for b in box], indexing='ij'), -1).reshape(-1, 2)
        uv_new.append(grid)
        uv_old.append(grid)
    edge_rows, outside = [], 0.
    new_occ = _occurrences(new)
    for (wire, old_edge), (_, edge) in zip(plan['occurrences'], new_occ):
        c_old, f0, l0 = BRep_Tool.CurveOnSurface(old_edge, original)
        c_new, f1, l1 = BRep_Tool.CurveOnSurface(edge, new)
        t = np.linspace(f1, l1, options.curve_samples)
        a = np.array([c_new.Value(float(x)).Coord() for x in t])
        lo, hi = (np.array([0., 0.]), np.array([1., 1.])) if g is not None else (box[:, 0], box[:, 1])
        outside = max(outside, float(np.max(np.r_[lo-a.min(0), a.max(0)-hi])))
        spatial = BRepAdaptor_Curve(edge)
        q3 = np.array([spatial.Value(float(x)).Coord() for x in t])
        qs, _ = d1(new_s, a)
        old_t = np.linspace(f0, l0, options.curve_samples)
        q_old = np.array([BRepAdaptor_Curve(old_edge).Value(float(x)).Coord() for x in old_t])
        b = np.array([c_old.Value(float(x)).Coord() for x in old_t])
        qo, _ = d1(old_s, b)
        edge_rows.append({'wire': wire, 'new_pcurve_vs_3d_max_mm': float(np.linalg.norm(qs-q3, axis=1).max()),
                          'old_pcurve_vs_3d_max_mm': float(np.linalg.norm(qo-q_old, axis=1).max()),
                          'edge_tolerance_old': float(BRep_Tool.Tolerance(old_edge)),
                          'edge_tolerance_new': float(BRep_Tool.Tolerance(edge))})
        # Trim samples in the new chart, mapped back, also test the surface.
        uv_new.append(a)
        uv_old.append(to_native(a) if g is not None else a)
    uv_new, uv_old = np.vstack(uv_new), np.vstack(uv_old)
    p_new, n_new = d1(new_s, uv_new)
    p_old, n_old = d1(old_s, uv_old)
    position = float(np.linalg.norm(p_new-p_old, axis=1).max())
    scale = np.linalg.norm(n_new, axis=1)*np.linalg.norm(n_old, axis=1)
    cosine = np.einsum('ij,ij->i', n_new, n_old)/np.maximum(scale, np.finfo(float).tiny)
    regular = float(np.linalg.norm(n_new, axis=1).min())
    worst_edge = max(r['new_pcurve_vs_3d_max_mm']-max(r['old_pcurve_vs_3d_max_mm'], r['edge_tolerance_old'])
                     for r in edge_rows)
    tol_growth = max(r['edge_tolerance_new']-r['edge_tolerance_old'] for r in edge_rows)
    checks = {
        'surface_position_max_mm': position,
        'normal_cosine_min': float(cosine.min()),
        'jacobian_normal_min': regular,
        'pcurve_outside_new_domain_max': outside,
        'pcurve_vs_3d_excess_max_mm': float(worst_edge),
        'edge_tolerance_growth_max_mm': float(tol_growth),
        'samples': int(len(uv_new)), 'edges': edge_rows,
    }
    failures = []
    if not position <= options.geometry_tolerance:
        failures.append('surface moved %.3g mm' % position)
    if not cosine.min() > 1-1e-6:
        failures.append('normal orientation changed (cos %.6g)' % cosine.min())
    if not regular > 0:
        failures.append('degenerate Jacobian in new chart')
    if not outside <= 1e-9:
        failures.append('pcurve leaves new UV domain by %.3g' % outside)
    if not worst_edge <= options.edge_tolerance:
        failures.append('pcurve/3D curve mismatch %.3g mm' % worst_edge)
    if not tol_growth <= options.edge_tolerance:
        failures.append('edge tolerance grew by %.3g mm' % tol_growth)
    checks['passed'] = not failures
    checks['failures'] = failures
    return checks


def _build(shape, plans, options):
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Copy
    copy = BRepBuilderAPI_Copy(shape, True).Shape()
    originals, faces = _faces(shape), _faces(copy)
    if len(faces) != len(originals):
        raise RuntimeError('shape copy changed the face count')
    checks = {}
    for index, plan in plans.items():
        _apply(faces[index], plan, _new_pcurves(plan))
    for index, plan in plans.items():
        checks[index] = _check_face(originals[index], faces[index], plan, options)
    return copy, checks


def _compare_status(before, after, options):
    failures = []
    if after['counts'] != before['counts']:
        failures.append('topology counts changed: %s -> %s' % (before['counts'], after['counts']))
    for key in ('topologically_valid', 'boolean_valid'):
        if before[key] and not after[key]:
            failures.append(key+' lost')
    if after['n_closed_shells'] < before['n_closed_shells']:
        failures.append('closed shells lost')
    for key in ('volume', 'area'):
        if not abs(after[key]-before[key]) <= options.volume_rtol*max(abs(before[key]), 1.):
            failures.append('%s changed: %.12g -> %.12g' % (key, before[key], after[key]))
    return failures


def reparameterize_shape(shape, options=None):
    """Return ``(new_shape, report)``; faces that fail any check keep their chart."""
    options = options or StepUVOptions()
    originals = _faces(shape)
    rows, plans = [], {}
    for index, face in enumerate(originals):
        try:
            row, plan = _plan_face(index, face, options)
        except (ValueError, RuntimeError) as error:
            row, plan = {'face': index, 'action': 'unchanged', 'reason': 'planning failed: %s' % error}, None
        rows.append(row)
        if plan is not None:
            plans[index] = plan
    before = shape_status(shape)
    rejected = {}
    while True:
        new, checks = _build(shape, plans, options)
        failed = [i for i, c in checks.items() if not c['passed']]
        for i in failed:
            rejected[i] = checks[i]
            plans.pop(i)
        if not failed:
            break
    for row in rows:
        i = row['face']
        if i in checks:
            row['checks'] = checks[i]
        elif i in rejected:
            row.update({'checks': rejected[i], 'action': 'unchanged',
                        'reason': 'reverted: ' + '; '.join(rejected[i]['failures']), 'rejected_action': row['action']})
    after = shape_status(new) if plans else before
    failures = _compare_status(before, after, options)
    report = {'schema_version': SCHEMA, 'options': _jsonable(asdict(options)),
              'faces': rows, 'before': before, 'after': after, 'failures': failures,
              'changed_faces': sorted(plans), 'reverted_faces': sorted(rejected),
              'actions': {a: sum(r['action'] == a for r in rows) for a in ('strip', 'restrict', 'unchanged')},
              'passed': not failures}
    return new, report


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_step(path):
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_Reader
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise RuntimeError('cannot read STEP file %s' % path)
    reader.TransferRoots()
    return reader.OneShape()


def reparameterize_step(input_step, output_step, report_path, options=None, on_failure='fallback'):
    """Write a geometry-identical STEP with automatic UV charts and a JSON report.

    If the reparameterized shape or its STEP readback fails a global check,
    ``on_failure='fallback'`` writes the input file unchanged (status
    ``fallback_original``); ``'error'`` raises instead. Either way the report
    records every check.
    """
    import shutil
    from cadmap.conversion.export import write_step
    if on_failure not in ('fallback', 'error'):
        raise ValueError("on_failure must be 'fallback' or 'error'")
    options = options or StepUVOptions()
    input_step, output_step, report_path = Path(input_step), Path(output_step), Path(report_path)
    output_step.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    shape = read_step(input_step)
    new, report = reparameterize_shape(shape, options)
    report.update({'input_step': str(input_step), 'input_sha256': _sha256(input_step)})
    if report['passed'] and report['changed_faces']:
        temporary = output_step.with_name(output_step.stem+'.candidate'+output_step.suffix)
        write_step(new, str(temporary))
        readback = read_step(temporary)
        report['readback'] = shape_status(readback)
        failures = _compare_status(report['before'], report['readback'], options)
        areas = np.abs(_face_areas(readback)-_face_areas(shape))
        report['readback']['face_area_difference_max'] = float(areas.max())
        if not areas.max() <= options.volume_rtol*max(1., float(_face_areas(shape).max())):
            failures.append('face areas changed after STEP readback')
        report['readback']['failures'] = failures
        report['passed'] = not failures
        if report['passed']:
            temporary.replace(output_step)
        else:
            temporary.unlink()
    if not report['passed']:
        if on_failure == 'error':
            _write_json(report_path, {**report, 'status': 'failed'})
            raise RuntimeError('UV reparameterization failed its checks; inspect %s' % report_path)
        shutil.copyfile(input_step, output_step)
        report['status'] = 'fallback_original'
    else:
        if not report['changed_faces']:
            shutil.copyfile(input_step, output_step)
        report['status'] = 'reparameterized' if report['changed_faces'] else 'unchanged'
    report.update({'output_step': str(output_step), 'output_sha256': _sha256(output_step)})
    _write_json(report_path, report)
    return report


def _write_json(path, report):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(_jsonable(report), indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)
