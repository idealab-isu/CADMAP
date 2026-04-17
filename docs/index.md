# CADMap

**CADMap** is a Python library for topology-preserving deformation of CAD (STEP) models to match pre-segmented point clouds.

## Overview

Given a CAD solid and a set of point-cloud patches (one per surface), CADMap deforms the NURBS control points via differentiable optimization while enforcing C0 continuity along all shared edges, keeping the model water-tight throughout.

## Key Features

- NURBS geometry classes with differentiable evaluation via NURBSDiff
- OpenCASCADE (STEP) import/export
- Topology-preserving constraint matrix (null-space gradient projection)
- Chamfer and Hausdorff distance metrics
- Hungarian-algorithm surface-to-patch matching
- Open3D and Matplotlib visualization helpers

## Quick Start

```python
from cadmap.conversion.occ import SolidConvertor, add_geom_to_solid, create_diffsurf_of_solid
from cadmap.sampling.pointcloud import sample_points_from_solid
from cadmap.deformation.fitting import CADMapOptimizer

# Load a STEP file and build the topology tree
solid = SolidConvertor("my_part.step").convert()
add_geom_to_solid(solid)
create_diffsurf_of_solid(solid, device="cpu")

# Sample evaluation points from each surface
cad_pts = sample_points_from_solid(solid, n_pts=500)

# Run the optimizer (targets = list of point-cloud arrays, one per face)
optimizer = CADMapOptimizer(solid, targets=my_targets, device="cpu", n_epochs=500)
snapshots = optimizer.run()

# Export the result
result_solid = optimizer.export_occ_solid()
```

## Installation

```bash
pip install -e ".[dev]"
```

See [Installation](installation.md) for details on manual dependencies (pythonocc-core, NURBSDiff, pytorch3d).

## API Reference

- [Geometry](api/geometry.md) — `BsplineSurface`, `Curve2d`, `Curve3d`, `Topology`
- [Conversion](api/conversion.md) — `SolidConvertor` and OCC helpers
- [Sampling](api/sampling.md) — `sample_points_from_solid`
- [Deformation](api/deformation.md) — `CADMapOptimizer`, constraint matrix
- [Registration](api/registration.md) — Surface matching and geometric inversion
- [Metrics](api/metrics.md) — Chamfer and Hausdorff distances
- [Visualization](api/visualization.md) — Plotting and Open3D apps
