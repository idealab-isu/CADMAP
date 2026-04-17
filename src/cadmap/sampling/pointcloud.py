"""Point-cloud generation from STEP files and Topology objects."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepTopAdaptor import BRepTopAdaptor_FClass2d
from OCC.Core.gp import gp_Pnt2d
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.Precision import precision
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_IN
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from OCC.Core.BRepTools import breptools

from cadmap.geometry.topology import Topology
from cadmap._internal.math_ops import batch_wn


# ---------------------------------------------------------------------------
# Raw file I/O
# ---------------------------------------------------------------------------

def extract_pc(dataFileName: str, sample_rate: float = 1.0) -> np.ndarray:
    """Load a tab-delimited point-cloud file.

    Parameters
    ----------
    dataFileName:
        Path to the text file (columns: x y z [r g b …]).
    sample_rate:
        Fraction of points to keep (uniform stride, 0 < sample_rate ≤ 1).

    Returns
    -------
    np.ndarray of shape (N, 3).
    """
    target = np.genfromtxt(dataFileName, delimiter="\t", dtype=np.float32)
    if target.ndim == 1:
        target = target.reshape(1, -1)
    if target.shape[1] > 3:
        target = target[:, :3]
    stride = max(1, int(1 / sample_rate))
    return target[np.arange(0, target.shape[0], stride)]


# ---------------------------------------------------------------------------
# Sampling from a cadmap Topology solid
# ---------------------------------------------------------------------------

def sample_from_solid(
    solid: Topology,
    missing_rate: float = 0.2,
    uv_sample_size: Tuple[int, int] = (24, 24),
) -> List[np.ndarray]:
    """Sample 3-D points from every face of a cadmap Topology solid.

    Parameters
    ----------
    solid:
        Root topology node (faces are ``solid.descendent``).
    missing_rate:
        Fraction of in-trim points to randomly drop (simulates missing data).
    uv_sample_size:
        Grid resolution ``(nu, nv)`` in parametric space.

    Returns
    -------
    List of (M, 3) arrays, one per face.
    """
    all_points: List[np.ndarray] = []
    for face in solid.descendent:
        surface = face.geom
        u, v = np.meshgrid(
            np.linspace(0, 1, uv_sample_size[0]),
            np.linspace(0, 1, uv_sample_size[1]),
            indexing="ij",
        )
        points2d = np.stack([u, v], axis=-1)
        points3d = np.array(surface.evaluate_list(surface.init_ctp, points2d.reshape(-1, 2)))
        intrim = batch_wn(points2d, face)
        points3d = points3d[intrim.reshape(-1)]
        np.random.shuffle(points3d)
        keep = int(points3d.shape[0] * (1 - missing_rate))
        all_points.append(points3d[:keep])
    return all_points


# ---------------------------------------------------------------------------
# Sampling directly from an OCC TopoDS_Shape (no cadmap Topology required)
# ---------------------------------------------------------------------------

def compute_face_size(face) -> float:
    """Approximate face size via its axis-aligned bounding box diagonal."""
    bbox = Bnd_Box()
    brepbndlib.Add(face, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    return max(xmax - xmin, ymax - ymin, zmax - zmin)


def distribute_samples_among_faces(
    occ_solid,
    total_samples: int = 10_000,
) -> dict:
    """Distribute *total_samples* proportionally across all faces of *occ_solid*."""
    sizes, faces = [], []
    it = TopExp_Explorer(occ_solid, TopAbs_FACE)
    while it.More():
        f = topods.Face(it.Current())
        sizes.append(compute_face_size(f))
        faces.append(f)
        it.Next()

    total_size = sum(sizes)
    if total_size == 0:
        raise ValueError("All faces have zero area.")
    n_per_face = [max(1, int(s / total_size * total_samples)) for s in sizes]
    return dict(zip(faces, n_per_face))


def sample_points_from_trimmed_face(
    face,
    num_samples: int,
) -> Optional[np.ndarray]:
    """Randomly sample *num_samples* points inside the trimmed region of *face*.

    Returns an (M, 3) array (M ≤ num_samples) or ``None`` if no valid points found.
    """
    surface = BRep_Tool.Surface(face)
    u_min, u_max, v_min, v_max = breptools.UVBounds(face)
    classifier = BRepTopAdaptor_FClass2d(face, precision.Confusion())

    u_vals = np.random.uniform(u_min, u_max, num_samples)
    v_vals = np.random.uniform(v_min, v_max, num_samples)

    pts = []
    for u, v in zip(u_vals, v_vals):
        state = classifier.Perform(gp_Pnt2d(u, v))
        if state == TopAbs_IN:
            props = GeomLProp_SLProps(surface, u, v, 1, 1e-6)
            if props.IsNormalDefined():
                p = props.Value()
                pts.append([p.X(), p.Y(), p.Z()])

    return np.array(pts) if pts else None


def sample_points_from_solid(
    occ_solid,
    total_samples: int = 10_000,
) -> List[np.ndarray]:
    """Sample ~*total_samples* points from all faces of an OCC solid.

    Parameters
    ----------
    occ_solid:
        A ``TopoDS_Solid`` (or any ``TopoDS_Shape`` with faces).
    total_samples:
        Approximate total number of output points.

    Returns
    -------
    List of (M, 3) arrays, one per face.
    """
    sampled: List[np.ndarray] = []
    dist = distribute_samples_among_faces(occ_solid, total_samples)
    for face, n in dist.items():
        pts = sample_points_from_trimmed_face(face, n)
        if pts is not None:
            sampled.append(pts)
    total = sum(len(f) for f in sampled)
    print(f"Sampled {len(sampled)} faces with {total} total points.")
    return sampled


# ---------------------------------------------------------------------------
# Convenience: generate a point cloud directly from a STEP file
# ---------------------------------------------------------------------------

def generate_point_cloud_from_step(
    filename_path: str,
    uv_sample_size: Tuple[int, int] = (40, 40),
) -> List[np.ndarray]:
    """Load a STEP file and sample points from it.

    Uses the cadmap :class:`~cadmap.geometry.topology.Topology` pipeline, which
    normalises surfaces to [0, 1] × [0, 1] parametric space.
    """
    from OCC.Extend.DataExchange import read_step_file
    from cadmap.conversion.occ import SolidConvertor, add_geom_to_solid

    shp = read_step_file(filename_path)
    solid = Topology(shp)
    SolidConvertor(solid).convert()
    add_geom_to_solid(solid)
    return sample_from_solid(solid, uv_sample_size=uv_sample_size)
