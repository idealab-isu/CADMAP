"""Validated paired mesh archive loading, preserving the existing frame convention."""
from pathlib import Path
import hashlib

import numpy as np


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_displacement_field(modal_npz, fit_npz, iteration=-1):
    """Return nodes, displacement, triangles, and CAD-to-measurement rigid pose.

    These trusted local research archives contain pickled diagnostic data.
    Node ordering is an input contract. Shape and pose validation detect many,
    but not all, correspondence mistakes; provenance remains necessary.
    """
    from cadmap.deformation.mesh_transfer import rigid_transform

    with np.load(modal_npz, allow_pickle=True) as modal:
        nodes = np.asarray(modal["Coordinates"], dtype=np.float64)
        raw_triangles = np.asarray(modal["Elements"])
    with np.load(fit_npz, allow_pickle=True) as fit:
        init = np.asarray(fit["initLoc"], dtype=np.float64)
        final = np.asarray(fit["CurrentLoc_list"][iteration], dtype=np.float64)
    if nodes.ndim != 2 or nodes.shape[1] != 3 or len(nodes) < 3:
        raise ValueError("source nodes must have shape (N,3), N >= 3")
    if init.shape != nodes.shape or final.shape != nodes.shape:
        raise ValueError("source/initial/final node arrays must have matching shape")
    if not all(np.isfinite(a).all() for a in (nodes, init, final, raw_triangles)):
        raise ValueError("mesh arrays contain nonfinite data")
    triangles = raw_triangles.astype(np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles):
        raise ValueError("triangles must have shape (M,3), M > 0")
    if not np.array_equal(triangles, raw_triangles) or triangles.min() < 0 or triangles.max() >= len(nodes):
        raise ValueError("triangle connectivity must be integral zero-based node indices")
    if np.any(np.diff(np.sort(triangles, axis=1), axis=1) == 0):
        raise ValueError("triangle has repeated vertex indices")
    transform = rigid_transform(nodes, init)
    residual = np.linalg.norm(transform.apply(nodes)-init, axis=1)
    scale = max(float(np.linalg.norm(np.ptp(nodes, axis=0))), 1.)
    if residual.max() > 1e-5 * scale:
        raise ValueError("source/init node pairing is inconsistent with a rigid frame change")
    displacement = transform.inverse().apply_vectors(final-init)
    return nodes, displacement, triangles, transform
