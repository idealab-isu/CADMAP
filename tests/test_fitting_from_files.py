"""Integration test: CADMap fitting from STEP + multi-cloud PLY files.

Expected layout in tests/fixtures/:
  bar.step                          — as-designed STEP model
  pointcloud/bar/bar_bend.ply       — bent bar point clouds (field: cloud_id)
  pointcloud/bar/bar_twist.ply      — twisted bar point clouds (field: cloud_id)

The PLY files must contain per-vertex fields: x, y, z, cloud_id.
Each unique cloud_id corresponds to one surface face in the STEP file.

Run with:
    pytest tests/test_fitting_from_files.py -v -s
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

FIXTURES      = Path(__file__).parent / "fixtures"
BAR_STEP      = FIXTURES / "step" / "bar.step"
BAR_BEND_PLY  = FIXTURES / "pointcloud" / "bar" / "bar_bend.ply"
BAR_TWIST_PLY = FIXTURES / "pointcloud" / "bar" / "bar_twist.ply"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_ply_patches(
    ply_path: Path,
    max_pts: int | None = None,
    seed: int = 42,
) -> List[np.ndarray]:
    """Read a multi-cloud PLY file and return a list of (N_i, 3) arrays.

    The PLY must contain vertex fields ``x``, ``y``, ``z``, and ``cloud_id``.
    The returned list is ordered by ascending ``cloud_id``.

    Parameters
    ----------
    max_pts:
        If set, randomly subsample each patch to at most *max_pts* points.
        Use this to keep test runtime manageable on CPU.
    seed:
        RNG seed for reproducible subsampling.
    """
    from plyfile import PlyData

    plydata = PlyData.read(str(ply_path))
    v = plydata["vertex"]

    xyz      = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    cloud_id = np.array(v["cloud_id"])

    rng = np.random.default_rng(seed)
    patches = []
    for cid in sorted(np.unique(cloud_id)):
        patch = xyz[cloud_id == cid]
        if max_pts is not None and len(patch) > max_pts:
            idx = rng.choice(len(patch), max_pts, replace=False)
            patch = patch[idx]
        print(f"  Cloud {cid}: {patch.shape[0]} points")
        patches.append(patch)

    return patches


def _load_solid(step_path: Path):
    """Load a STEP file and return a fully prepared Topology solid."""
    from OCC.Extend.DataExchange import read_step_file
    from cadmap.geometry.topology import Topology
    from cadmap.conversion.occ import (
        SolidConvertor,
        add_geom_to_solid,
        create_diffsurf_of_solid,
        compute_inside_evalpts,
    )

    shp = read_step_file(str(step_path))
    solid = Topology(shp)
    SolidConvertor(solid).convert()
    add_geom_to_solid(solid)
    create_diffsurf_of_solid(solid)
    compute_inside_evalpts(solid)
    return solid


def _run_fitting(solid, targets: List[np.ndarray], n_epochs: int = 100):
    """Run CADMapOptimizer and return the optimizer instance."""
    from cadmap.deformation.fitting import CADMapOptimizer

    optimizer = CADMapOptimizer(
        solid,
        targets=targets,
        device="cpu",
        lr=10.0,
        n_epochs=n_epochs,
        sample_size=150,
    )
    optimizer.run()
    return optimizer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not BAR_STEP.exists(),
    reason=f"{BAR_STEP} not found",
)
@pytest.mark.skipif(
    not BAR_BEND_PLY.exists(),
    reason=f"{BAR_BEND_PLY} not found",
)
def test_bar_bend_fitting():
    """Fit the as-designed bar to the bent point cloud."""
    print(f"\nLoading STEP: {BAR_STEP.name}")
    solid = _load_solid(BAR_STEP)

    print(f"Loading PLY:  {BAR_BEND_PLY.name}")
    targets = _read_ply_patches(BAR_BEND_PLY, max_pts=1500)

    n_faces   = len(solid.descendent)
    n_patches = len(targets)
    print(f"Faces: {n_faces}  |  PLY clouds: {n_patches}")
    assert n_faces == n_patches, (
        f"STEP has {n_faces} faces but PLY has {n_patches} clouds — "
        "one cloud_id per face is required."
    )

    opt = _run_fitting(solid, targets)

    print(f"Initial loss : {opt.loss_history[0]:.6f}")
    print(f"Final loss   : {opt.loss_history[-1]:.6f}")
    assert opt.loss_history[-1] < opt.loss_history[0], "Loss did not decrease."


@pytest.mark.skipif(
    not BAR_STEP.exists(),
    reason=f"{BAR_STEP} not found",
)
@pytest.mark.skipif(
    not BAR_TWIST_PLY.exists(),
    reason=f"{BAR_TWIST_PLY} not found",
)
def test_bar_twist_fitting():
    """Fit the as-designed bar to the twisted point cloud."""
    print(f"\nLoading STEP: {BAR_STEP.name}")
    solid = _load_solid(BAR_STEP)

    print(f"Loading PLY:  {BAR_TWIST_PLY.name}")
    targets = _read_ply_patches(BAR_TWIST_PLY, max_pts=1500)

    n_faces   = len(solid.descendent)
    n_patches = len(targets)
    print(f"Faces: {n_faces}  |  PLY clouds: {n_patches}")
    assert n_faces == n_patches, (
        f"STEP has {n_faces} faces but PLY has {n_patches} clouds."
    )

    opt = _run_fitting(solid, targets)

    print(f"Initial loss : {opt.loss_history[0]:.6f}")
    print(f"Final loss   : {opt.loss_history[-1]:.6f}")
    assert opt.loss_history[-1] < opt.loss_history[0], "Loss did not decrease."


@pytest.mark.skipif(
    not BAR_STEP.exists() or not BAR_BEND_PLY.exists(),
    reason="bar.step or bar_bend.ply not found",
)
def test_bar_bend_export():
    """Fitted solid can be exported as a valid STEP file."""
    import os, tempfile
    from cadmap.visualization.plots import write_step

    solid   = _load_solid(BAR_STEP)
    targets = _read_ply_patches(BAR_BEND_PLY, max_pts=1500)

    opt = _run_fitting(solid, targets, n_epochs=10)

    occ_solid = opt.export_occ_solid()
    assert occ_solid is not None

    with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as f:
        tmp_path = f.name
    try:
        write_step(occ_solid, tmp_path)
        size = os.path.getsize(tmp_path)
        assert size > 0, "Exported STEP file is empty."
        print(f"\nExported: {tmp_path} ({size} bytes)")
    finally:
        os.unlink(tmp_path)
