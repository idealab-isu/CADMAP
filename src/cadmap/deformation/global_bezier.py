"""Single-span trivariate Bezier displacement maps in physical coordinates.

The map is ``T(x) = x + sum B_abc(x) delta_D_abc``.  These are field
coefficients, not CAD surface control points.  No CAD approximation occurs here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.linalg import lstsq, svdvals


def _points(value):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all():
        raise ValueError("points must be a finite (N, 3) array")
    return a


def _degrees(value):
    if len(value) != 3 or any(int(v) != v or v < 0 for v in value):
        raise ValueError("degrees must be three nonnegative integers")
    return tuple(int(v) for v in value)


def _bounds(value):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (2, 3) or not np.isfinite(a).all() or np.any(a[1] <= a[0]):
        raise ValueError("bounds must have finite, strictly positive extents")
    return a


def bernstein(t, degree):
    """Stable recurrence, including endpoints; deliberately never clips t."""
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    if degree < 0 or int(degree) != degree:
        raise ValueError("degree must be a nonnegative integer")
    out = np.ones((len(t), 1))
    for _ in range(int(degree)):
        nxt = np.zeros((len(t), out.shape[1] + 1))
        nxt[:, :-1] += (1 - t[:, None]) * out
        nxt[:, 1:] += t[:, None] * out
        out = nxt
    return out


def basis_matrix(points, bounds, degrees):
    x, box, deg = _points(points), _bounds(bounds), _degrees(degrees)
    uvw = (x - box[0]) / (box[1] - box[0])
    b = [bernstein(uvw[:, i], deg[i]) for i in range(3)]
    return np.einsum("ni,nj,nk->nijk", *b).reshape(len(x), int(np.prod(np.array(deg) + 1)))


def second_difference_operator(degrees):
    """Parameter-space curvature penalty, normalized by sqrt(number of rows).

    Each axis uses p(p-1) times consecutive coefficient second differences.
    Affine displacements lie in the nullspace.  This is a lattice penalty,
    not an integral of physical-space curvature.
    """
    deg = _degrees(degrees)
    shape = tuple(d + 1 for d in deg)
    ids = np.arange(np.prod(shape)).reshape(shape)
    rows = []
    for axis, degree in enumerate(deg):
        if degree < 2:
            continue
        seq = np.moveaxis(ids, axis, -1).reshape(-1, degree + 1)
        for line in seq:
            for k in range(degree - 1):
                row = np.zeros(ids.size)
                row[line[k:k + 3]] = degree * (degree - 1) * np.array([1., -2., 1.])
                rows.append(row)
    return np.asarray(rows).reshape(-1, ids.size) / np.sqrt(max(1, len(rows)))


@dataclass
class BezierDeformation:
    bounds: np.ndarray
    degrees: Tuple[int, int, int]
    coefficients: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.bounds = _bounds(self.bounds).copy()
        self.degrees = _degrees(self.degrees)
        shape = tuple(d + 1 for d in self.degrees) + (3,)
        a = np.asarray(self.coefficients, dtype=np.float64)
        if a.size != np.prod(shape) or not np.isfinite(a).all():
            raise ValueError("invalid displacement coefficients")
        self.coefficients = a.reshape(shape).copy()

    def outside(self, points, tolerance=1e-12):
        x = _points(points)
        q = (x - self.bounds[0]) / np.diff(self.bounds, axis=0)[0]
        return np.any((q < -tolerance) | (q > 1 + tolerance), axis=1)

    def displacement(self, points, batch_size=8192):
        x = _points(points)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        out = np.empty_like(x)
        for lo in range(0, len(x), batch_size):
            sl = slice(lo, lo + batch_size)
            out[sl] = basis_matrix(x[sl], self.bounds, self.degrees) @ self.coefficients.reshape(-1, 3)
        return out

    def evaluate(self, points, batch_size=8192):
        x = _points(points)
        return x + self.displacement(x, batch_size)

    def jacobian(self, points, batch_size=8192):
        """Return J[n, output_coordinate, input_coordinate], including identity."""
        x = _points(points)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        out = np.broadcast_to(np.eye(3), (len(x), 3, 3)).copy()
        lengths = self.bounds[1] - self.bounds[0]
        for axis, degree in enumerate(self.degrees):
            if not degree:
                continue
            deg = list(self.degrees)
            deg[axis] -= 1
            derivative = (degree / lengths[axis]) * np.diff(self.coefficients, axis=axis)
            for lo in range(0, len(x), batch_size):
                sl = slice(lo, lo + batch_size)
                out[sl, :, axis] += basis_matrix(x[sl], self.bounds, deg) @ derivative.reshape(-1, 3)
        return out

    def save(self, path):
        np.savez_compressed(path, bounds=self.bounds, degrees=self.degrees,
                            coefficients=self.coefficients,
                            metadata=np.array(json.dumps(self.metadata)), version=np.array(1))

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as z:
            if int(z["version"]) != 1:
                raise ValueError("unsupported field version")
            return cls(z["bounds"], tuple(z["degrees"]), z["coefficients"],
                       json.loads(str(z["metadata"])))


def fit_bezier(points, displacement, degrees, bounds, regularization=0., weights=None,
               rcond=1e-12):
    """SVD least squares with normalized data weights and an augmented penalty.

    Objective: sum_i w_i ||A_i C-U_i||^2 / sum(w) + lambda ||L C||^2.
    Rank of the unregularized data matrix is always reported separately.
    """
    x, u = _points(points), _points(displacement)
    if len(x) == 0 or x.shape != u.shape:
        raise ValueError("nonempty paired points and displacement must have equal shape")
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and nonnegative")
    if not np.isfinite(rcond) or rcond <= 0:
        raise ValueError("rcond must be finite and positive")
    w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (len(x),) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be finite, nonnegative, with positive sum")
    a = basis_matrix(x, bounds, degrees)
    root = np.sqrt(w / w.sum())[:, None]
    aw, rhs = a * root, u * root
    spectrum = svdvals(aw)
    rank = int(np.sum(spectrum > rcond * spectrum[0]))
    condition = float(spectrum[0] / spectrum[-1]) if rank == a.shape[1] else None
    penalty = second_difference_operator(degrees)
    design = aw
    if regularization and len(penalty):
        design = np.vstack([aw, np.sqrt(regularization) * penalty])
        rhs = np.vstack([rhs, np.zeros((len(penalty), 3))])
    c, _, augmented_rank, _ = lstsq(design, rhs, cond=rcond, lapack_driver="gelsd")
    model = BezierDeformation(bounds, degrees, c)
    model.metadata = {
        "n_data": len(x), "n_coefficients": a.shape[1], "regularization": regularization,
        "data_rank": rank, "augmented_rank": int(augmented_rank), "rcond": rcond,
        "data_condition": condition, "singular_values": spectrum.tolist(),
        "max_coefficient_norm": float(np.linalg.norm(c, axis=1).max()),
        "training_rms": float(np.sqrt(np.mean(np.sum((a @ c-u)**2, axis=1)))),
        "penalty": "p(p-1) parameter-axis second differences / sqrt(total rows)",
        "training_outside_box": int(model.outside(x).sum()),
    }
    return model


def error_statistics(error, weights=None):
    a = np.asarray(error, dtype=float)
    values = np.linalg.norm(a, axis=1) if a.ndim == 2 else a
    if not len(values):
        return {"count": 0, "mean": None, "rms": None, "p95": None, "max": None}
    if not np.isfinite(values).all():
        raise ValueError("nonfinite error")
    w = np.ones(len(values)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != values.shape or np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("invalid metric weights")
    order = np.argsort(values)
    p95 = values[order[np.searchsorted(np.cumsum(w[order]), .95*w.sum())]]
    return {"count": len(values), "mean": float(np.average(values, weights=w)),
            "rms": float(np.sqrt(np.average(values**2, weights=w))),
            "p95": float(p95), "max": float(values.max())}


def validation_split(points, seed=17):
    """Freeze test nodes, spatial-block validation, and random validation.

    The final test set never participates in hyperparameter selection.  Block
    labels use source geometry only; test nodes can share blocks with training.
    """
    x = _points(points)
    if len(x) < 50:
        raise ValueError("at least 50 nodes needed for the four-way split")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x))
    test = perm[:max(1, len(x)//10)]
    dev = perm[len(test):]
    extent = np.ptp(x, axis=0)
    norm = (x-x.min(axis=0)) / np.where(extent > 0, extent, 1.)
    cells = np.minimum((norm * [6, 4, 2]).astype(int), [5, 3, 1])
    ids = np.ravel_multi_index(cells.T, (6, 4, 2))
    cell_order = rng.permutation(np.unique(ids[dev]))
    if len(cell_order) < 2:
        raise ValueError("source geometry has too few occupied spatial blocks")
    block_ids = cell_order[:max(1, len(cell_order)//5)]
    block = dev[np.isin(ids[dev], block_ids)]
    pool = dev[~np.isin(ids[dev], block_ids)]
    random = pool[:max(1, len(pool)//5)]
    train = pool[len(random):]
    if not all(len(v) for v in (test, block, random, train)):
        raise ValueError("empty validation subset")
    return {"train": np.sort(train), "random_validation": np.sort(random),
            "spatial_validation": np.sort(block), "test": np.sort(test),
            "block_ids": ids, "heldout_blocks": block_ids}
