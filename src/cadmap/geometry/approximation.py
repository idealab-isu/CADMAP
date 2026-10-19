"""Conservative Bernstein degree reduction and OCC-compatible spline assembly.

Polynomial reductions use coefficient-space least squares and a convex-hull
error bound. Positive-weight rational pieces already within the degree limit
are retained exactly. This is a CADMap approximation, not Hube's Algorithm 2.
"""
from functools import lru_cache
from itertools import product

import numpy as np
from scipy.linalg import lstsq

from cadmap.geometry.composition import (
    RationalBezierPiece, PiecewiseRationalSpline, elevate,
)


@lru_cache(maxsize=128)
def elevation_matrix(low, high):
    return elevate(np.eye(low + 1), [high])


def _reduce_axis(net, degree, axis):
    data = np.moveaxis(net, axis, 0)
    current = data.shape[0] - 1
    if current == degree:
        return net.copy()
    e = elevation_matrix(degree, current)
    # Fix both endpoints of every control-net line. In particular, adjoining
    # curve pieces retain their source endpoints without subsequent averaging.
    result = np.empty((degree + 1,) + data.shape[1:])
    result[0], result[-1] = data[0], data[-1]
    rhs = data - e[:, :1].reshape((-1,) + (1,) * (data.ndim-1)) * data[0]
    rhs -= e[:, -1:].reshape((-1,) + (1,) * (data.ndim-1)) * data[-1]
    if degree > 1:
        result[1:-1] = lstsq(e[:, 1:-1], rhs.reshape(current+1, -1), cond=1e-13)[0].reshape(result[1:-1].shape)
    return np.moveaxis(result, 0, axis)


def _elevate_axis(net, degree, axis):
    data = np.moveaxis(net, axis, 0)
    result = np.tensordot(elevation_matrix(data.shape[0]-1, degree), data, axes=(1, 0))
    return np.moveaxis(result, 0, axis)


def reduce_polynomial_piece(piece, degrees):
    """Return a reduced polynomial piece and a conservative position error bound.

    A near-constant denominator (roundoff from polynomial composition) is
    explicitly accounted for in the bound. General rational reduction is not
    silently approximated by moving dehomogenized control points.
    """
    h = piece.homogeneous
    w = h[..., -1]
    if np.ptp(w) > 64*np.finfo(float).eps*w.max():
        raise ValueError("rational degree reduction requires a separate weighted approximation")
    mean_w = float(w.mean())
    origin = h.reshape(-1, 4)[0, :3] / w.flat[0]
    centered = (h[..., :3] - w[..., None]*origin) / mean_w
    reduced = centered
    for axis, degree in enumerate(degrees):
        reduced = _reduce_axis(reduced, degree, axis)
    elevated = reduced
    for axis, degree in enumerate(piece.degrees):
        elevated = _elevate_axis(elevated, degree, axis)
    radius = np.linalg.norm(reduced, axis=-1).max()
    bound = (mean_w*np.linalg.norm(elevated-centered, axis=-1).max()
             + radius*np.abs(w-mean_w).max()) / w.min()
    result = np.concatenate([reduced+origin, np.ones(reduced.shape[:-1]+(1,))], axis=-1)
    return RationalBezierPiece(result, piece.intervals), float(bound)


def reduce_rational_piece(piece, degrees):
    """Reduce homogeneous coordinates, bounding the induced rational error."""
    h=piece.homogeneous.copy();w=h[...,-1]
    origin=h.reshape(-1,4)[0,:3]/w.flat[0]
    h[...,:3]-=w[...,None]*origin
    low=h
    for axis,degree in enumerate(degrees):low=_reduce_axis(low,degree,axis)
    if np.any(low[...,-1]<=0):return None,float('inf')
    high=low
    for axis,degree in enumerate(piece.degrees):high=_elevate_axis(high,degree,axis)
    radius=np.linalg.norm(low[...,:3]/low[...,-1:],axis=-1).max()
    bound=(np.linalg.norm(high[...,:3]-h[...,:3],axis=-1).max()
           +radius*np.abs(high[...,-1]-w).max())/w.min()
    result=low.copy();result[...,:3]+=result[...,-1:]*origin
    return RationalBezierPiece(result,piece.intervals),float(bound)


def approximate_spline(spline, tolerance=1e-9, max_degree=25):
    """Choose the smallest tested common degree meeting a coefficient error bound."""
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("positive tolerance required")
    if max_degree < 1 or max_degree > 25:
        raise ValueError("OCC degree limit must be between 1 and 25")
    original = spline.pieces[0].degrees
    if any(p.degrees != original for p in spline.pieces):
        raise ValueError("all source pieces must share their representation degrees")
    rational = any(np.ptp(p.homogeneous[..., -1]) > 64*np.finfo(float).eps*p.homogeneous[..., -1].max()
                   for p in spline.pieces)
    if rational:
        if max(original) > max_degree:
            raise ValueError("rational geometry exceeds OCC limit; weighted reduction required")
        selected=original;pieces=spline.pieces;error=0.
        for axis in range(len(original)):
            for degree in [1,2,3,4,6,8,10,12,16,20,25]:
                if degree>=selected[axis]:break
                trial=list(selected);trial[axis]=degree
                candidates=[reduce_rational_piece(p,trial) for p in spline.pieces]
                bound=max(b for _,b in candidates)
                if bound<=tolerance:
                    selected=tuple(trial);pieces=[p for p,_ in candidates];error=bound;break
        method='homogeneous rational reduction' if selected!=original else 'exact rational retention'
        info={"degrees":list(selected),"coefficient_error_bound":error,"method":method}
        meta=dict(spline.metadata);meta['approximation']=info
        return PiecewiseRationalSpline(pieces,meta),info
    candidates = ([(3,), (4,), (6,), (8,), (10,), (12,), (16,), (20,), (25,)]
                  if len(original) == 1 else
                  [(4, 4), (8, 6), (10, 8), (12, 8), (14, 10), (16, 12),
                   (18, 14), (20, 16), (22, 18), (24, 20), (25, 22), (25, 25)])
    seen, last = set(), None
    for candidate in candidates:
        degrees = tuple(min(d, o, max_degree) for d, o in zip(candidate, original))
        if degrees in seen:
            continue
        seen.add(degrees)
        pieces, bounds = [], []
        for p in spline.pieces:
            reduced, bound = reduce_polynomial_piece(p, degrees)
            pieces.append(reduced)
            bounds.append(bound)
            if bound > tolerance:
                break
        last = max(bounds)
        if len(pieces) == len(spline.pieces) and last <= tolerance:
            meta = dict(spline.metadata)
            meta["approximation"] = {"degrees": list(degrees), "coefficient_error_bound": last,
                                     "method": "endpoint-constrained Bernstein coefficient least squares"}
            return PiecewiseRationalSpline(pieces, meta), meta["approximation"]
    raise ValueError(f"degree budget cannot meet approximation tolerance {tolerance}: last bound {last}")


def assemble_bspline(spline, join_tolerance=1e-9):
    """Assemble Cartesian pieces into one C0 BSpline without moving their geometry.

    Homogeneous scales are reconciled at joins. A non-proportional rational
    interface is rejected. Control discrepancies at roundoff scale are recorded.
    Returns Euclidean poles, weights, expanded knots, degrees, and join error.
    """
    degrees = spline.pieces[0].degrees
    if any(p.degrees != degrees for p in spline.pieces):
        raise ValueError("piece degrees differ")
    shape = tuple((len(b)-2)*d+d+1 for b,d in zip(spline.breaks,degrees))
    h = np.full(shape+(4,), np.nan)
    worst = 0.
    for idx in product(*[range(len(b)-1) for b in spline.breaks]):
        p = spline.pieces[spline._lookup[idx]]
        values = p.homogeneous.copy()
        slices = tuple(slice(k*d, (k+1)*d+1) for k,d in zip(idx,degrees))
        dest = h[slices]
        known = np.isfinite(dest[..., -1])
        if known.any():
            scale = float(np.median(dest[..., -1][known]/values[..., -1][known]))
            values *= scale
            old_xyz = dest[known, :3]/dest[known, -1:]
            new_xyz = values[known, :3]/values[known, -1:]
            err = float(np.linalg.norm(new_xyz-old_xyz, axis=1).max())
            weight_error = float(np.abs(values[..., -1][known]-dest[..., -1][known]).max())
            if err > join_tolerance or weight_error > 1e-11*float(values[..., -1].max()):
                raise ValueError(f"incompatible Bezier join: position={err}, weight={weight_error}")
            worst = max(worst, err)
        dest[~known] = values[~known]
    knots = [np.repeat(b, [d+1]+[d]*(len(b)-2)+[d+1]) for b,d in zip(spline.breaks,degrees)]
    return h[..., :3]/h[..., -1:], h[..., -1], knots, degrees, worst


def occ_geometry(spline):
    """Construct a supported OCC curve/surface and report any join roundoff."""
    cp, weights, knots, degrees, join = assemble_bspline(spline)
    if max(degrees) > 25:
        raise ValueError("geometry exceeds OCC maximum degree")
    if len(degrees) == 2:
        from cadmap.geometry.nurbs import BsplineSurface
        surf = BsplineSurface(cp, knots[0], knots[1], surface_id="practical", ctp_start_index=0,
                              periodic_u=False, periodic_v=False, weight=weights,
                              degree_u=degrees[0], degree_v=degrees[1])
        return surf._to_geomsurface(), join
    from OCC.Core.Geom import Geom_BSplineCurve
    from OCC.Core.TColgp import TColgp_Array1OfPnt
    from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
    from OCC.Core.gp import gp_Pnt
    pts, w = TColgp_Array1OfPnt(1,len(cp)), TColStd_Array1OfReal(1,len(cp))
    for i,(point,weight) in enumerate(zip(cp,weights),1):
        pts.SetValue(i,gp_Pnt(*map(float,point)))
        w.SetValue(i,float(weight))
    unique, counts = np.unique(knots[0],return_counts=True)
    kv, mult = TColStd_Array1OfReal(1,len(unique)), TColStd_Array1OfInteger(1,len(unique))
    for i,(k,m) in enumerate(zip(unique,counts),1):
        kv.SetValue(i,float(k)); mult.SetValue(i,int(m))
    return Geom_BSplineCurve(pts,w,kv,mult,degrees[0],False), join
