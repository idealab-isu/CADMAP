"""Automatic trim-aware selection of simple buffered support charts.

Boundary fitting proposes a reference polynomial; it does not certify coverage.
Exact inverse pcurves and subdivided rational Bernstein hulls bound every trim
occurrence, including holes and seams. A modest occupancy target stops the
search from rewarding tight or unnecessarily high-degree support boundaries.
"""
from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib
import numpy as np
from scipy.optimize import linprog

from cadmap.deformation.global_bezier import bernstein
from cadmap.geometry.composition import RationalBezierPiece, PiecewiseRationalSpline
from cadmap.geometry.uv_reparameterization import PolynomialStripUVMap, support_regularity


@dataclass(frozen=True)
class TrimUVOptions:
    trigger_occupancy: float = .20
    satisfactory_occupancy: float = .20
    minimum_area_gain: float = 2.
    transverse_buffer_fraction: float = .10
    longitudinal_buffer_fraction: float = .002
    bound_tolerance: float = 1e-7
    max_degree: int = 3
    max_exact_coefficients: int = 100000
    max_condition: float = 250.

    def __post_init__(self):
        values = [v for v in asdict(self).values()]
        if (not np.isfinite(values).all() or not 0 < self.trigger_occupancy < 1
                or not self.trigger_occupancy <= self.satisfactory_occupancy <= 1
                or self.minimum_area_gain <= 1 or self.transverse_buffer_fraction <= 0
                or self.longitudinal_buffer_fraction <= 0 or self.bound_tolerance <= 0
                or int(self.max_degree) != self.max_degree or not 0 <= self.max_degree <= 3
                or self.max_exact_coefficients <= 0 or self.max_condition <= 1):
            raise ValueError('invalid automatic UV options')


def _split(h):
    left, right = [h[0]], [h[-1]]
    while len(h) > 1:
        h = .5*(h[:-1]+h[1:])
        left.append(h[0]); right.append(h[-1])
    return np.array(left), np.array(right[::-1])


def curve_bounds(curves, tolerance=1e-7, max_depth=12):
    """Conservative hull enclosure, even when the refinement budget is reached.

    This is a float64 algebraic enclosure with an explicit roundoff allowance,
    not interval arithmetic or a proof about the STEP translator.
    """
    lo = np.full(2, np.inf); hi = -lo; leaves = 0; max_used = 0
    for curve in curves:
        for p in curve.pieces:
            pending = [(p.homogeneous, 0)]
            while pending:
                h, depth = pending.pop()
                cp = h[:, :2]/h[:, -1:]
                low, high = cp.min(0), cp.max(0)
                ends = cp[[0, -1]]
                excess = max(float((ends.min(0)-low).max()), float((high-ends.max(0)).max()))
                if excess > tolerance and depth < max_depth:
                    a, b = _split(h)
                    pending.extend([(a, depth+1), (b, depth+1)])
                else:
                    lo = np.minimum(lo, low); hi = np.maximum(hi, high)
                    leaves += 1; max_used = max(max_used, depth)
    if not leaves:
        raise ValueError('trim has no pcurve pieces')
    guard = 512*np.finfo(float).eps*max(1., float(np.abs(np.r_[lo, hi]).max()))
    return lo-guard, hi+guard, {'method': 'subdivided positive-weight rational Bernstein hulls',
                              'leaves': leaves, 'maximum_depth': max_used,
                              'roundoff_allowance': guard, 'formal_interval_certificate': False}


def sample_trims(curves, count=5):
    return np.concatenate([p.evaluate(np.linspace(*p.intervals[0], count))
                           for curve in curves for p in curve.pieces])


def trim_area(curves):
    """Signed Green integral using already wire-oriented imported pcurves.

    Numerical quadrature is a diagnostic only; coverage uses curve hulls.
    """
    nodes, weights = np.polynomial.legendre.leggauss(12)
    wires = defaultdict(float)
    for curve in curves:
        for p in curve.pieces:
            a, b = p.intervals[0]
            t = a+(nodes+1)*(b-a)/2
            xy, tangent = p.evaluate(t, derivatives=True)
            integral = np.dot(weights, xy[:, 0]*tangent[:, 1, 0]-xy[:, 1]*tangent[:, 0, 0])*(b-a)/4
            wires[str(curve.metadata.get('wire', 0))] += float(integral)
    return abs(sum(wires.values())), dict(wires)


def face_fingerprint(surface, curves):
    """Geometry identity independent of face names and occurrence enumeration."""
    hasher = hashlib.sha256()
    for p in surface.pieces:
        hasher.update(p.homogeneous.tobytes()); hasher.update(p.intervals.tobytes())
    parts = []
    for curve in curves:
        forward = b''.join(p.homogeneous.tobytes()+p.intervals.tobytes() for p in curve.pieces)
        parts.append(hashlib.sha256(forward).digest())
    for part in sorted(parts):
        hasher.update(part)
    return hasher.hexdigest()


def _source_invariance(surface, transformed, mapping, curves, rng):
    # Boundary points plus an independent grid/random support check. Physical
    # equality is algebraic everywhere; the samples measure floating-point error.
    old = sample_trims(curves, 7)
    uv = np.vstack([mapping.to_practical(old), rng.random((257, 2))])
    original = mapping.to_original(uv)
    a, da = surface.evaluate(original, derivatives=True, extrapolate=True)
    b, db = transformed.evaluate(uv, derivatives=True, extrapolate=True)
    chain = np.einsum('nij,njk->nik', da, mapping.jacobian(uv))
    scale = np.maximum(1., np.linalg.norm(chain.reshape(len(uv), -1), axis=1))
    pos = float(np.linalg.norm(a-b, axis=1).max())
    derivative = float((np.linalg.norm((chain-db).reshape(len(uv), -1), axis=1)/scale).max())
    inverse = float(np.abs(mapping.to_practical(original)-uv).max())
    return {'passed': pos <= 1e-9 and derivative <= 1e-9 and inverse <= 1e-10,
            'sampled_position_max_mm': pos, 'sampled_scaled_derivative_max': derivative,
            'sampled_inverse_max': inverse, 'samples': len(uv),
            'scope': 'exact homogeneous construction with independent floating-point checks; samples are not a domain certificate'}


def generate_candidates(surface, curves, options=None, field_degree_sum=10):
    options = options or TrimUVOptions()
    points = sample_trims(curves)
    low, high, _ = curve_bounds(curves, options.bound_tolerance)
    area, wires = trim_area(curves)
    rows = []; accepted = []
    for axis in (0, 1):
        extent = high[axis]-low[axis]
        if extent <= 100*options.bound_tolerance:
            continue
        margin = options.longitudinal_buffer_fraction*extent
        bounds = (float(low[axis]-margin), float(high[axis]+margin))
        x = (points[:, axis]-bounds[0])/(bounds[1]-bounds[0])
        y = points[:, 1-axis]
        for degree in range(options.max_degree+1):
            row = {'axis': axis, 'degree': degree, 'accepted': False}
            rows.append(row)
            try:
                if len(surface.pieces) != 1:
                    raise ValueError('unsupported exact path: multiple source spans')
                if not np.allclose(surface.pieces[0].intervals, [[0, 1], [0, 1]], rtol=0, atol=1e-12):
                    raise ValueError('automatic diagnostics require a normalized source UV domain')
                basis = bernstein(x, degree)
                # A minimax strip supplies only a reference shape. Buffered
                # widths, meaningful improvement and complexity decide selection.
                constraints = np.vstack([np.column_stack((basis, np.zeros(len(x)))),
                                         np.column_stack((-basis, -np.ones(len(x))))])
                solution = linprog(np.r_[np.zeros(degree+1), 1.], A_ub=constraints,
                                   b_ub=np.r_[y, -y], bounds=[(None, None)]*(degree+1)+[(0, None)], method='highs')
                if not solution.success:
                    raise ValueError('reference fit failed: '+solution.message)
                reference = solution.x[:-1]
                residual_map = PolynomialStripUVMap(axis, tuple(reference), 1., bounds)
                residual_curves = [residual_map.inverse_pcurve(c) for c in curves]
                rlo, rhi, residual_bound = curve_bounds(residual_curves, options.bound_tolerance)
                # Undo the axis=1 orientation reflection to obtain y-f(x).
                a, b = ((float(rlo[1]), float(rhi[1])) if axis == 0
                        else (float(1-rhi[1]), float(1-rlo[1])))
                span = b-a
                if span <= 100*options.bound_tolerance:
                    raise ValueError('near-zero trim width')
                buffer = max(options.transverse_buffer_fraction*span, 10*options.bound_tolerance)
                mapping = PolynomialStripUVMap(axis, tuple(reference+a-buffer), span+2*buffer, bounds)
                inverse_curves = [mapping.inverse_pcurve(c) for c in curves]
                clo, chi, coverage = curve_bounds(inverse_curves, options.bound_tolerance)
                margin_new = np.minimum(clo, 1-chi)
                row.update({'map': mapping.metadata(), 'support_area': mapping.determinant,
                            'occupancy': float(area/mapping.determinant),
                            'area_gain': float(1/mapping.determinant),
                            'transverse_buffer_old_uv': float(buffer),
                            'longitudinal_buffer_old_uv': float(margin),
                            'coverage': {**coverage, 'bounds': [clo.tolist(), chi.tolist()],
                                         'minimum_margin_new_uv': margin_new.tolist(),
                                         'occurrences': len(curves)},
                            'residual_bounds': residual_bound})
                if np.any(margin_new <= 0):
                    raise ValueError('complete trim coverage with positive buffer not verified')
                source_degree = np.array(surface.pieces[0].degrees)
                p, q = source_degree[axis], source_degree[1-axis]
                predicted = np.array([p+degree*q, q])*max(1, field_degree_sum)
                row['predicted_exact_degrees'] = predicted.tolist()
                row['predicted_exact_coefficients'] = int(np.prod(predicted+1))
                if row['predicted_exact_coefficients'] > options.max_exact_coefficients:
                    raise ValueError('deformed exact coefficient budget exceeded')
                if np.ptp(surface.pieces[0].homogeneous[..., -1]) > 1e-12 and max(predicted) > 25:
                    raise ValueError('unsupported OCC path: high-degree rational reduction required')
                if row['area_gain'] < options.minimum_area_gain:
                    raise ValueError('buffered support does not provide meaningful area improvement')
                uv = np.column_stack((np.linspace(0, 1, 129), np.full(129, .5)))
                condition = np.linalg.cond(mapping.jacobian(uv))
                row['sampled_uv_condition_max'] = float(condition.max())
                slope_bound = float(np.max(np.abs(degree*np.diff(mapping.coefficients)))) if degree else 0.
                condition_bound = float(np.linalg.cond([[mapping.length, 0], [slope_bound, mapping.width]]))
                row['uv_condition_bernstein_upper_bound'] = condition_bound
                if condition_bound > options.max_condition:
                    raise ValueError('UV condition budget exceeded')
                transformed = mapping.reparameterize_source(surface, options.max_exact_coefficients)
                row['support_regularity'] = support_regularity(transformed)
                if not row['support_regularity']['passed']:
                    raise ValueError('source continuation regularity not certified by normal-cone bound')
                row['source_invariance'] = _source_invariance(surface, transformed, mapping, curves, np.random.default_rng(421))
                if not row['source_invariance']['passed']:
                    raise ValueError('source invariance check failed')
                row['accepted'] = True
                row['satisfactory_occupancy'] = row['occupancy'] >= options.satisfactory_occupancy
                # Once useful occupancy is achieved, extra tightness earns no
                # reward. Lower algebraic cost wins; buffer is already enforced.
                row['rank'] = [0 if row['satisfactory_occupancy'] else 1,
                               row['predicted_exact_coefficients'], degree,
                               0. if row['satisfactory_occupancy'] else -row['occupancy'], axis]
                accepted.append((mapping, row))
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                row['rejection_reason'] = str(error)
    accepted.sort(key=lambda item: item[1]['rank'])
    return accepted, rows


def select_automatic_maps(sources, options=None, field_degree_sum=10):
    """Scan all source faces; IDs label results but never drive eligibility."""
    options = options or TrimUVOptions()
    pcurves = defaultdict(list)
    for name, spline in sources.items():
        if name.startswith('pcurve_'):
            pcurves['face_'+str(spline.metadata['face'])].append(spline)
    maps, report, exact_sources, exact_pcurves = {}, [], {}, {}
    for key, surface in sources.items():
        if not key.startswith('face_'):
            continue
        curves = pcurves[key]
        if not curves:
            report.append({'face': key, 'selected': False, 'reason': 'missing trim occurrences'})
            continue
        if not np.allclose([[b[0], b[-1]] for b in surface.breaks], [[0, 1], [0, 1]], rtol=0, atol=1e-12):
            report.append({'face': key, 'selected': False,
                           'reason': 'unsupported: automatic diagnostics require normalized source UV'})
            continue
        area, wires = trim_area(curves)
        low, high, enclosure = curve_bounds(curves, options.bound_tolerance)
        row = {'face': key, 'fingerprint': face_fingerprint(surface, curves),
               'trim_area_quadrature': float(area), 'wire_signed_areas': wires,
               'original_occupancy': float(area), 'trim_bounds': [low.tolist(), high.tolist()],
               'original_bounds_method': enclosure, 'source_spans': len(surface.pieces),
               'source_degrees': list(surface.pieces[0].degrees), 'pcurve_occurrences': len(curves),
               'selected': False}
        report.append(row)
        if area <= 0 or area > 1+1e-6:
            row['reason'] = 'invalid oriented trim-area diagnostic; automatic selection disabled'
        elif area >= options.trigger_occupancy:
            row['reason'] = 'existing support occupancy sufficient'
        else:
            accepted, candidates = generate_candidates(surface, curves, options, field_degree_sum)
            row['candidates'] = candidates
            if not accepted:
                row['reason'] = 'unsupported: no feasible simple buffered improvement'
                continue
            mapping, chosen = accepted[0]
            maps[key] = mapping
            exact_sources[key] = mapping.reparameterize_source(surface)
            for curve in curves:
                exact_pcurves[curve.metadata['pcurve']] = mapping.inverse_pcurve(curve)
            row.update({'selected': True, 'map': mapping.metadata(), 'selected_candidate': chosen,
                        'reason': 'simplest feasible candidate after satisfactory occupancy preference'})
    return maps, {'version': 1, 'options': asdict(options), 'faces': report,
                  'selection_uses_face_ids': False, 'manual_coefficients_used': False,
                  'kernel_acceptance_pending': True,
                  'objective': 'simple support with positive buffer and useful occupancy; no reward for tightness beyond the occupancy target'}, exact_sources, exact_pcurves
