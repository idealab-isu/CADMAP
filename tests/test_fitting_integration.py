"""Integration test: full CADMap fitting pipeline on real STEP files.

Place STEP files in tests/fixtures/ before running:
  - bar.stp   — simple 6-face bar geometry (untrimmed NURBS)
  - cube.step — axis-aligned box (planar faces, fastest smoke test)

Run only these (slow) tests with:
    pytest tests/test_fitting_integration.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_and_prepare(step_path: Path, device: str = "cpu"):
    """Load a STEP file and run the full pre-optimisation setup."""
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


def _sample_targets(solid, missing_rate: float = 0.0):
    """Sample point-cloud targets from the solid's own surfaces (self-fit test).

    Uses each surface's own eval grid so the shape matches ``inside_evalpts``.
    """
    targets = []
    for face in solid.descendent:
        surf = face.geom
        u, v = np.meshgrid(
            np.linspace(0, 1, surf.uEvalPtSize),
            np.linspace(0, 1, surf.vEvalPtSize),
            indexing="ij",
        )
        pts2d = np.stack([u, v], axis=-1)
        pts3d = np.array(surf.evaluate_list(surf.init_ctp, pts2d.reshape(-1, 2)))
        mask = surf.inside_evalpts.reshape(-1)
        pts3d = pts3d[mask]
        if missing_rate > 0 and len(pts3d) > 0:
            keep = max(1, int(len(pts3d) * (1 - missing_rate)))
            pts3d = pts3d[np.random.choice(len(pts3d), keep, replace=False)]
        targets.append(pts3d)
    return targets


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (FIXTURES / "cube.step").exists(),
    reason="tests/fixtures/cube.step not found",
)
def test_smoke_cube():
    """Smoke test: pipeline runs without error on a simple box geometry."""
    from cadmap.deformation.fitting import CADMapOptimizer

    solid = _load_and_prepare(FIXTURES / "cube.step")
    targets = _sample_targets(solid)

    optimizer = CADMapOptimizer(
        solid,
        targets=targets,
        device="cpu",
        lr=1.0,
        n_epochs=3,
        sample_size=5,
    )
    snapshots = optimizer.run()

    assert len(snapshots) == 4  # initial + 3 epochs
    assert optimizer.loss_history[0] >= 0


@pytest.mark.skipif(
    not (FIXTURES / "bar.stp").exists(),
    reason="tests/fixtures/bar.stp not found",
)
def test_self_fit_bar():
    """Self-fit: fitting a bar to its own point cloud should give near-zero loss."""
    from cadmap.deformation.fitting import CADMapOptimizer

    solid = _load_and_prepare(FIXTURES / "bar.stp")
    targets = _sample_targets(solid)

    optimizer = CADMapOptimizer(
        solid,
        targets=targets,
        device="cpu",
        lr=1.0,
        n_epochs=5,
        sample_size=8,
    )
    optimizer.run()

    # Self-fitting should converge to near zero
    final_loss = optimizer.loss_history[-1]
    assert final_loss < 0.1, f"Self-fit loss did not converge: {final_loss:.4f}"


@pytest.mark.skipif(
    not (FIXTURES / "bar.stp").exists(),
    reason="tests/fixtures/bar.stp not found",
)
def test_loss_decreases_over_epochs():
    """Loss should decrease (or stay stable) over training epochs."""
    from cadmap.deformation.fitting import CADMapOptimizer

    solid = _load_and_prepare(FIXTURES / "bar.stp")

    # Add slight jitter to targets to create a non-trivial fitting problem
    targets = _sample_targets(solid, missing_rate=0.2)
    targets = [t + np.random.normal(0, 0.5, t.shape) for t in targets]

    optimizer = CADMapOptimizer(
        solid,
        targets=targets,
        device="cpu",
        lr=5.0,
        n_epochs=10,
        sample_size=8,
    )
    optimizer.run()

    history = optimizer.loss_history
    assert len(history) == 10
    # Loss at the end should not be worse than at the start
    assert history[-1] <= history[0] * 1.5, (
        f"Loss increased significantly: {history[0]:.4f} → {history[-1]:.4f}"
    )


@pytest.mark.skipif(
    not (FIXTURES / "bar.stp").exists(),
    reason="tests/fixtures/bar.stp not found",
)
def test_export_occ_solid():
    """Fitted solid can be exported as a valid OCC solid."""
    from cadmap.deformation.fitting import CADMapOptimizer
    from cadmap.visualization.plots import write_step
    import tempfile, os

    solid = _load_and_prepare(FIXTURES / "bar.stp")
    targets = _sample_targets(solid)

    optimizer = CADMapOptimizer(
        solid, targets=targets, device="cpu", lr=1.0, n_epochs=2, sample_size=5
    )
    optimizer.run()

    occ_solid = optimizer.export_occ_solid()
    assert occ_solid is not None

    with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as f:
        tmp_path = f.name
    try:
        write_step(occ_solid, tmp_path)
        assert os.path.getsize(tmp_path) > 0
    finally:
        os.unlink(tmp_path)
