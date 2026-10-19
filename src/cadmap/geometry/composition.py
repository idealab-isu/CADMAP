"""Algebraic composition with a polynomial Bezier displacement field.

High-degree rational Bezier pieces are stored independently of OCC. Knot
insertion, Bernstein products and homogeneous de Casteljau implement exact
spline operations (subject to floating-point roundoff), without sampling/refits.
Piece boundaries subdivide parameter space, not the original B-Rep topology.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
import json
from math import comb

import numpy as np
from scipy.interpolate import BSpline

from cadmap.deformation.global_bezier import bernstein


@lru_cache(maxsize=128)
def _binomial_grid(shape):
    out = np.ones(shape)
    for axis, size in enumerate(shape):
        view = [1] * len(shape)
        view[axis] = size
        out *= np.array([comb(size - 1, i) for i in range(size)], float).reshape(view)
    if not np.isfinite(out).all():
        raise ValueError("degree exceeds float64 Bernstein product capacity")
    out.setflags(write=False)
    return out


def bernstein_product(a, b):
    """Multiply tensor Bernstein polynomials of one or two variables.

    ``b`` is scalar; ``a`` may have one trailing component axis. Products use
    binomial-scaled convolution, evaluated by shifts of the smaller scalar net.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    dim = b.ndim
    if dim not in (1, 2) or a.ndim not in (dim, dim + 1):
        raise ValueError("expected curve/surface polynomials and optional components")
    if a.ndim == dim and a.size < b.size:
        a, b = b, a
    ashape = a.shape[:dim]
    shape = tuple(x + y - 1 for x, y in zip(ashape, b.shape))
    tail = (None,) if a.ndim == dim + 1 else ()
    aa = a * _binomial_grid(ashape)[(...,) + tail]
    bb = b * _binomial_grid(b.shape)
    out = np.zeros(shape + a.shape[dim:])
    for idx in np.ndindex(b.shape):
        slices = tuple(slice(i, i + size) for i, size in zip(idx, ashape))
        out[slices] += bb[idx] * aa
    out /= _binomial_grid(shape)[(...,) + tail]
    return out


def elevate(coefficients, degrees):
    """Degree elevation by multiplication with the constant polynomial one."""
    a = np.asarray(coefficients, float)
    dim = len(degrees)
    delta = np.asarray(degrees) - (np.array(a.shape[:dim]) - 1)
    if np.any(delta < 0) or np.any(delta != delta.astype(int)):
        raise ValueError("target degrees must be integers at least the current degrees")
    return bernstein_product(a, np.ones(tuple(delta.astype(int) + 1)))


def _polynomial_value(coefficients, parameters):
    bases = [bernstein(parameters[:, i], n - 1)
             for i, n in enumerate(coefficients.shape[:-1])]
    if len(bases) == 1:
        return bases[0] @ coefficients
    return np.einsum("ni,nj,ijc->nc", *bases, coefficients, optimize=True)


@dataclass
class RationalBezierPiece:
    homogeneous: np.ndarray
    intervals: np.ndarray

    def __post_init__(self):
        self.homogeneous = np.asarray(self.homogeneous, float).copy()
        self.intervals = np.asarray(self.intervals, float).copy()
        dim = self.homogeneous.ndim - 1
        if dim not in (1, 2) or self.homogeneous.shape[-1] not in (3, 4):
            raise ValueError("expected homogeneous 2D/3D curve or surface coefficients")
        if (self.intervals.shape != (dim, 2) or not np.isfinite(self.intervals).all()
                or np.any(np.diff(self.intervals, axis=1) <= 0)):
            raise ValueError("invalid parameter intervals")
        if not np.isfinite(self.homogeneous).all() or np.any(self.homogeneous[..., -1] <= 0):
            raise ValueError("homogeneous coefficients require finite positive weights")

    @property
    def degrees(self):
        return tuple(n - 1 for n in self.homogeneous.shape[:-1])

    def evaluate(self, parameters, derivatives=False, batch_size=512, extrapolate=False):
        """Return positions, optionally derivatives w.r.t. original parameters.

        Derivative array has shape (N, spatial dimension, parameter dimension).
        By default extrapolation is rejected and tiny endpoint roundoff is
        snapped to the end. Explicit extrapolation retains the parameters and
        evaluates the same rational polynomial continuation, without clipping.
        """
        p = np.asarray(parameters, float)
        if p.ndim == 1 and len(self.degrees) == 1:
            p = p[:, None]
        if p.ndim != 2 or p.shape[1] != len(self.degrees) or not np.isfinite(p).all():
            raise ValueError("invalid parameter array")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        widths = np.diff(self.intervals, axis=1)[:, 0]
        t = (p - self.intervals[:, 0]) / widths
        if not extrapolate and (np.any(t < -1e-12) or np.any(t > 1 + 1e-12)):
            raise ValueError("parameters outside piece domain")
        if not extrapolate:
            t = np.clip(t, 0, 1)
        xyz = np.empty((len(p), self.homogeneous.shape[-1] - 1))
        jac = np.empty(xyz.shape + (len(self.degrees),)) if derivatives else None
        for lo in range(0, len(p), batch_size):
            sl = slice(lo, lo + batch_size)
            h = _polynomial_value(self.homogeneous, t[sl])
            if not np.isfinite(h).all() or np.any(h[:, -1] <= 0):
                raise ValueError("nonfinite value or nonpositive rational denominator")
            xyz[sl] = h[:, :-1] / h[:, -1:]
            if derivatives:
                for axis, degree in enumerate(self.degrees):
                    if degree == 0:
                        jac[sl, :, axis] = 0
                        continue
                    dc = np.diff(self.homogeneous, axis=axis) * degree / widths[axis]
                    dh = _polynomial_value(dc, t[sl])
                    jac[sl, :, axis] = (dh[:, :-1] - xyz[sl] * dh[:, -1:]) / h[:, -1:]
        return (xyz, jac) if derivatives else xyz


@dataclass
class PiecewiseRationalSpline:
    pieces: list
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.pieces:
            raise ValueError("at least one Bezier piece is required")
        # A Cartesian partition gives unambiguous lookup and explicit adjacency.
        dim = len(self.pieces[0].degrees)
        spatial = self.pieces[0].homogeneous.shape[-1]
        if any(len(p.degrees) != dim or p.homogeneous.shape[-1] != spatial for p in self.pieces):
            raise ValueError("inconsistent piece dimensions")
        self.breaks = [np.unique([p.intervals[a] for p in self.pieces]) for a in range(dim)]
        self._lookup = {}
        for i, piece in enumerate(self.pieces):
            idx = tuple(int(np.searchsorted(self.breaks[a], piece.intervals[a, 0]))
                        for a in range(dim))
            if idx in self._lookup or any(self.breaks[a][k + 1] != piece.intervals[a, 1]
                                         for a, k in enumerate(idx)):
                raise ValueError("overlapping or non-Cartesian pieces")
            self._lookup[idx] = i
        if len(self._lookup) != np.prod([len(b) - 1 for b in self.breaks]):
            raise ValueError("incomplete parameter partition")

    @property
    def coefficient_count(self):
        return sum(p.homogeneous[..., -1].size for p in self.pieces)

    def evaluate(self, parameters, derivatives=False, batch_size=512, extrapolate=False):
        """Use the right-hand piece on internal knots; optional end-piece continuation."""
        p = np.asarray(parameters, float)
        if p.ndim == 1 and len(self.breaks) == 1:
            p = p[:, None]
        if p.ndim != 2 or p.shape[1] != len(self.breaks) or not np.isfinite(p).all():
            raise ValueError("invalid parameter array")
        ids = []
        for a, b in enumerate(self.breaks):
            eps = 1e-12 * (b[-1] - b[0])
            if not extrapolate and (np.any(p[:, a] < b[0] - eps) or np.any(p[:, a] > b[-1] + eps)):
                raise ValueError("parameters outside spline domain")
            ids.append(np.clip(np.searchsorted(b, p[:, a], side="right") - 1, 0, len(b) - 2))
        group = np.array([self._lookup[tuple(idx)] for idx in zip(*ids)], dtype=int)
        xyz = np.empty((len(p), self.pieces[0].homogeneous.shape[-1] - 1))
        jac = np.empty(xyz.shape + (len(self.breaks),)) if derivatives else None
        for i in np.unique(group):
            mask = group == i
            result = self.pieces[i].evaluate(p[mask], derivatives, batch_size, extrapolate)
            if derivatives:
                xyz[mask], jac[mask] = result
            else:
                xyz[mask] = result
        return (xyz, jac) if derivatives else xyz

    def adjacency(self):
        pairs = []
        for idx, i in self._lookup.items():
            for axis in range(len(idx)):
                neighbour = list(idx)
                neighbour[axis] += 1
                if tuple(neighbour) in self._lookup:
                    pairs.append({"pieces": [i, self._lookup[tuple(neighbour)]], "axis": axis})
        return pairs


def extract_bezier(control_points, weights, knots, degrees, metadata=None):
    """Split a clamped, nonperiodic rational spline by exact knot insertion.

    Inputs are Euclidean control points and expanded knots, one vector per
    parameter direction. Interior multiplicities may include discontinuities.
    """
    c, w = np.asarray(control_points, float), np.asarray(weights, float)
    dim = len(degrees)
    if dim not in (1, 2) or len(knots) != dim or c.shape != w.shape + (c.shape[-1],):
        raise ValueError("inconsistent spline dimensions")
    if w.ndim != dim or c.shape[-1] not in (2, 3) or not np.isfinite(c).all():
        raise ValueError("expected finite 2D/3D curve or surface")
    if not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("positive finite rational weights required")
    w = w / w.max()  # global homogeneous scaling preserves the spline
    h = np.concatenate([c * w[..., None], w[..., None]], axis=-1)
    refined = []
    for axis, (kv, degree) in enumerate(zip(knots, degrees)):
        kv = np.asarray(kv, float)
        if int(degree) != degree or degree < 1:
            raise ValueError("source spline degrees must be positive integers")
        degree = int(degree)
        if (kv.ndim != 1 or len(kv) != h.shape[axis] + degree + 1
                or not np.isfinite(kv).all() or np.any(np.diff(kv) < 0)):
            raise ValueError("invalid expanded knot vector")
        unique, counts = np.unique(kv, return_counts=True)
        if len(unique) < 2 or counts[0] != degree + 1 or counts[-1] != degree + 1:
            raise ValueError("source must be clamped and nonperiodic")
        if np.any(counts[1:-1] > degree + 1):
            raise ValueError("invalid interior multiplicity")
        bs = BSpline(kv, h, degree, axis=axis)
        for k, multiplicity in zip(unique[1:-1], counts[1:-1]):
            if multiplicity < degree:
                bs = bs.insert_knot(k, degree - int(multiplicity))
        h = np.moveaxis(bs.c, 0, axis)
        refined.append(bs.t)
    ranges = [[(k, kv[k], kv[k + 1]) for k in range(degree, len(kv) - degree - 1)
               if kv[k + 1] > kv[k]] for kv, degree in zip(refined, degrees)]
    pieces = []
    for span in product(*ranges):
        slices = tuple(slice(k - degree, k + 1) for (k, _, _), degree in zip(span, degrees))
        pieces.append(RationalBezierPiece(h[slices], [[lo, hi] for _, lo, hi in span]))
    return PiecewiseRationalSpline(pieces, dict(metadata or {}))


def compose_piece(piece, deformation):
    """Construct coefficients of T(S) with denominator W**sum(field degrees)."""
    if piece.homogeneous.shape[-1] != 4:
        raise ValueError("deformation composition requires 3D source geometry")
    h = piece.homogeneous / piece.homogeneous[..., -1].max()
    w = h[..., -1]
    dim = len(piece.degrees)
    s = sum(deformation.degrees)
    if s == 0:
        result = h.copy()
        result[..., :3] += w[..., None] * deformation.coefficients.reshape(3)
        return RationalBezierPiece(result, piece.intervals)
    q = [(h[..., a] - deformation.bounds[0, a] * w)
         / (deformation.bounds[1, a] - deformation.bounds[0, a]) for a in range(3)]
    # Homogeneous de Casteljau replaces each outer coordinate by Q/W. Each
    # recurrence multiplies by Q or W-Q, raising the common denominator once.
    nets = {idx: deformation.coefficients[idx].reshape((1,) * dim + (3,))
            for idx in np.ndindex(deformation.coefficients.shape[:-1])}
    shape = list(deformation.coefficients.shape[:-1])
    for axis in range(3):
        while shape[axis] > 1:
            shape[axis] -= 1
            nxt = {}
            for idx in np.ndindex(tuple(shape)):
                neighbour = list(idx)
                neighbour[axis] += 1
                nxt[idx] = (bernstein_product(nets[idx], w - q[axis])
                            + bernstein_product(nets[tuple(neighbour)], q[axis]))
            nets = nxt
    wp = np.ones((1,) * dim)
    for _ in range(s - 1):
        wp = bernstein_product(wp, w)
    numerator = nets[(0, 0, 0)] + bernstein_product(h[..., :3], wp)
    denominator = bernstein_product(wp, w)
    result = np.concatenate([numerator, denominator[..., None]], axis=-1)
    result /= denominator.max()
    return RationalBezierPiece(result, piece.intervals)


def compose_spline(source, deformation):
    meta = dict(source.metadata)
    meta.update({"operation": "T(S): identity plus Bezier displacement",
                 "field_degrees": list(deformation.degrees),
                 "source_degrees": [list(p.degrees) for p in source.pieces]})
    return PiecewiseRationalSpline([compose_piece(p, deformation) for p in source.pieces], meta)


def save_splines(path, splines, metadata=None):
    """Portable NPZ archive, JSON manifest, no pickle or OCC handles."""
    arrays, items = {}, {}
    for number, (name, spline) in enumerate(splines.items()):
        keys = []
        for pi, piece in enumerate(spline.pieces):
            key = f"s{number}_p{pi}"
            arrays[key] = piece.homogeneous
            arrays[key + "_intervals"] = piece.intervals
            keys.append(key)
        items[name] = {"pieces": keys, "metadata": spline.metadata,
                       "adjacency": spline.adjacency()}
    manifest = {"version": 1, "metadata": metadata or {}, "splines": items}
    np.savez_compressed(path, manifest=np.array(json.dumps(manifest)), **arrays)


def load_splines(path):
    with np.load(path, allow_pickle=False) as z:
        manifest = json.loads(str(z["manifest"]))
        if manifest["version"] != 1:
            raise ValueError("unsupported composition archive")
        splines = {name: PiecewiseRationalSpline(
            [RationalBezierPiece(z[key], z[key + "_intervals"]) for key in item["pieces"]],
            item["metadata"]) for name, item in manifest["splines"].items()}
    return splines, manifest["metadata"]
