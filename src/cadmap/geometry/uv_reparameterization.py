"""Simple, buffered, orientation-preserving polynomial strip charts.

The chart changes coordinates only. Homogeneous Bernstein substitution keeps
single-span rational surfaces and all rational pcurve occurrences algebraic.
Unsupported representations raise errors instead of silently fitting geometry.
"""
from dataclasses import dataclass
import numpy as np

from cadmap.deformation.global_bezier import bernstein
from cadmap.geometry.composition import (
    RationalBezierPiece, PiecewiseRationalSpline, bernstein_product, elevate,
    extract_bezier,
)


def _power(a, n):
    result = np.ones((1,) * a.ndim)
    for _ in range(n):
        result = bernstein_product(result, a)
    return result


def _homogeneous_polynomial(coefficients, numerator, denominator):
    nets = [np.array([v], float) for v in coefficients]
    while len(nets) > 1:
        nets = [bernstein_product(a, denominator - numerator)
                + bernstein_product(b, numerator) for a, b in zip(nets[:-1], nets[1:])]
    return nets[0]


@dataclass(frozen=True)
class PolynomialStripUVMap:
    """x follows old axis; transverse coordinate has constant positive width.

    axis=0: old=(lo+length*x, f(x)+width*y).
    axis=1: old=(f(x)+width*(1-y), lo+length*x).
    The reflection in axis=1 is essential: both charts have positive determinant.
    """
    axis: int
    coefficients: tuple
    width: float
    longitudinal_bounds: tuple = (0., 1.)

    def __post_init__(self):
        c = tuple(float(v) for v in self.coefficients)
        b = tuple(float(v) for v in self.longitudinal_bounds)
        if (self.axis not in (0, 1) or not 1 <= len(c) <= 4 or len(b) != 2
                or not np.isfinite([*c, *b, self.width]).all()
                or self.width <= 0 or b[1] <= b[0]):
            raise ValueError('invalid polynomial strip chart')
        object.__setattr__(self, 'coefficients', c)
        object.__setattr__(self, 'longitudinal_bounds', b)

    @property
    def degree(self):
        return len(self.coefficients) - 1

    @property
    def length(self):
        return self.longitudinal_bounds[1] - self.longitudinal_bounds[0]

    @property
    def determinant(self):
        return self.length * self.width

    def _reference(self, x):
        return bernstein(np.asarray(x).reshape(-1), self.degree) @ self.coefficients

    def to_original(self, uv):
        p = np.asarray(uv, float)
        if p.shape[-1] != 2 or not np.isfinite(p).all():
            raise ValueError('finite UV coordinates required')
        q = p.reshape(-1, 2)
        long = self.longitudinal_bounds[0] + self.length * q[:, 0]
        transverse = self._reference(q[:, 0]) + self.width * (q[:, 1] if self.axis == 0 else 1-q[:, 1])
        return np.column_stack((long, transverse) if self.axis == 0 else (transverse, long)).reshape(p.shape)

    def to_practical(self, uv):
        p = np.asarray(uv, float)
        if p.shape[-1] != 2 or not np.isfinite(p).all():
            raise ValueError('finite UV coordinates required')
        q = p.reshape(-1, 2)
        x = (q[:, self.axis] - self.longitudinal_bounds[0]) / self.length
        y = (q[:, 1-self.axis] - self._reference(x)) / self.width
        return np.column_stack((x, y if self.axis == 0 else 1-y)).reshape(p.shape)

    def jacobian(self, uv):
        p = np.asarray(uv, float)
        x = p.reshape(-1, 2)[:, 0]
        slope = (bernstein(x, self.degree-1) @ (self.degree*np.diff(self.coefficients))
                 if self.degree else np.zeros(len(x)))
        result = np.zeros((len(x), 2, 2))
        result[:, self.axis, 0] = self.length
        result[:, 1-self.axis, 0] = slope
        result[:, 1-self.axis, 1] = self.width * (1 if self.axis == 0 else -1)
        return result.reshape(p.shape[:-1] + (2, 2))

    def metadata(self):
        return {'type': 'polynomial_strip', 'version': 1, 'axis': self.axis,
                'coefficients': list(self.coefficients), 'width': float(self.width),
                'longitudinal_bounds': list(self.longitudinal_bounds),
                'jacobian_determinant': float(self.determinant)}

    def coordinate_nets(self):
        long = np.asarray(self.longitudinal_bounds)[:, None]
        c = np.asarray(self.coefficients)
        transverse = np.column_stack((c, c+self.width))
        if self.axis:
            transverse = transverse[:, ::-1]
        return (long, transverse) if self.axis == 0 else (transverse, long)

    def reparameterize_source(self, spline, max_coefficients=100000):
        if len(spline.pieces) != 1 or len(spline.pieces[0].degrees) != 2:
            raise ValueError('unsupported exact UV path: source must be a single surface span')
        piece = spline.pieces[0]
        q = [(net-piece.intervals[a, 0])/np.diff(piece.intervals[a])[0]
             for a, net in enumerate(self.coordinate_nets())]
        degrees = np.sum([np.array(net.shape)-1 for net, d in zip(q, piece.degrees)
                          for _ in range(d)], axis=0)
        if np.prod(degrees+1) > max_coefficients:
            raise ValueError('exact UV composition coefficient budget exceeded')
        nets = {idx: piece.homogeneous[idx].reshape(1, 1, -1)
                for idx in np.ndindex(piece.homogeneous.shape[:-1])}
        shape = list(piece.homogeneous.shape[:-1])
        for axis in range(2):
            while shape[axis] > 1:
                shape[axis] -= 1
                nxt = {}
                for idx in np.ndindex(tuple(shape)):
                    other = list(idx); other[axis] += 1
                    nxt[idx] = (bernstein_product(nets[idx], 1-q[axis])
                                + bernstein_product(nets[tuple(other)], q[axis]))
                nets = nxt
        result = RationalBezierPiece(nets[(0, 0)], np.array([[0., 1.], [0., 1.]]))
        return PiecewiseRationalSpline([result], {**spline.metadata, 'uv_map': self.metadata()})

    def reparameterize_bilinear(self, spline):
        # Compatibility with the existing reconstruction driver; no bilinear restriction.
        return self.reparameterize_source(spline)

    def inverse_pcurve(self, spline):
        pieces = []
        n = max(1, self.degree)
        for p in spline.pieces:
            if len(p.degrees) != 1 or p.homogeneous.shape[-1] != 3:
                raise ValueError('expected a rational 2D pcurve')
            h = p.homogeneous; w = h[:, -1]
            x = (h[:, self.axis]-self.longitudinal_bounds[0]*w)/self.length
            wp = _power(w, n-1)
            xx = bernstein_product(x, wp)
            ww = bernstein_product(w, wp)
            f = bernstein_product(_homogeneous_polynomial(self.coefficients, x, w), _power(w, n-self.degree))
            yy = (bernstein_product(h[:, 1-self.axis], wp)-f)/self.width
            if self.axis:
                yy = ww-yy
            pieces.append(RationalBezierPiece(np.column_stack((xx, yy, ww)), p.intervals))
        return PiecewiseRationalSpline(pieces, {**spline.metadata, 'uv_map': self.metadata()})

    def transform_pcurve(self, curve):
        from OCC.Core.Geom2d import Geom2d_BSplineCurve
        from cadmap.geometry.approximation import approximate_spline, assemble_bspline
        from cadmap.geometry.nurbs import Curve2d
        curve = Geom2d_BSplineCurve.DownCast(curve)
        cp = np.array([curve.Pole(i).Coord() for i in range(1, curve.NbPoles()+1)])
        weights = np.array([curve.Weight(i) for i in range(1, curve.NbPoles()+1)])
        knots = np.repeat([curve.Knot(i) for i in range(1, curve.NbKnots()+1)],
                          [curve.Multiplicity(i) for i in range(1, curve.NbKnots()+1)])
        source = extract_bezier(cp, weights, [knots], [curve.Degree()])
        exact = self.inverse_pcurve(source)
        # The established approximation module operates on homogeneous 3D data.
        lifted = PiecewiseRationalSpline([
            RationalBezierPiece(np.column_stack((p.homogeneous[:, :2],
                                                  np.zeros(len(p.homogeneous)), p.homogeneous[:, -1])), p.intervals)
            for p in exact.pieces], exact.metadata)
        reduced, info = approximate_spline(lifted, 1e-11)
        poles, weights, expanded, degrees, _ = assemble_bspline(reduced)
        result = Curve2d(poles[:, :2], expanded[0], 'automatic_uv', False,
                         weight=weights, degree=degrees[0])._to_Geom2d_BSplineCurve()
        return result, info


def uv_map_from_metadata(data):
    if data.get('type') == 'polynomial_strip' and data.get('version') == 1:
        return PolynomialStripUVMap(**{k: data[k] for k in
                                      ('axis', 'coefficients', 'width', 'longitudinal_bounds')})
    raise ValueError('unsupported UV map metadata')


def support_regularity(spline):
    """Sufficient Bernstein normal-cone bound; not a global injectivity proof.

    A positive dot product of the homogeneous normal numerator with a fixed
    reference direction on the entire square certifies a regular source chart.
    Conservative rejection is preferable to calling samples a domain proof.
    """
    p = spline.pieces[0]; h = p.homogeneous; w = h[..., -1]
    derivatives = []
    for axis in range(2):
        dh = np.diff(h, axis=axis)*p.degrees[axis]
        if not dh.size:
            raise ValueError('degenerate source support')
        derivatives.append(bernstein_product(dh[..., :3], w)
                           - bernstein_product(h[..., :3], dh[..., -1]))
    a, b = derivatives
    cross = np.stack([bernstein_product(a[..., j], b[..., k])
                      - bernstein_product(a[..., k], b[..., j])
                      for j, k in ((1, 2), (2, 0), (0, 1))], axis=-1)
    _, jac = p.evaluate(np.array([[.5, .5]]), derivatives=True)
    normal = np.cross(jac[0, :, 0], jac[0, :, 1])
    normal /= max(np.linalg.norm(normal), np.finfo(float).tiny)
    projected = cross @ normal
    guard = 256*np.finfo(float).eps*max(float(np.abs(cross).max()), np.finfo(float).tiny)
    lower = float(projected.min())
    return {'passed': bool(lower > guard), 'method': 'positive Bernstein normal-cone bound with floating-point guard',
            'normal_numerator_lower_bound': lower, 'roundoff_guard': guard,
            'minimum_denominator_control_weight': float(w.min()), 'global_injectivity_certified': False}
