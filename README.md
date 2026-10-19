# CADMap runtime repository

CADMap fits trimmed NURBS CAD models to segmented point clouds and initializes
CAD control points from MeshMap displacement fields through processtrak.

This copy retains the complete current Python package and original Git history.
Tests, fixtures, paper assets, development documentation, caches and previous
calculation outputs are omitted from this checkout. Keep experiment data and
results outside this repository, for example in the sibling ../experiment folder.

## Installation

CADMap depends on MeshMap (the `EigenFit` package), which brings limatix. See
`../INSTALL.md` for details:

```powershell
conda create -n mapping312 -c conda-forge python=3.12 pythonocc-core h5py -y
conda activate mapping312
cd C:\PhD\CNDE\Mapping
pip install -e ./limatix -e "./MeshMap[ansys]" -e ./CADMap --extra-index-url https://download.pytorch.org/whl/cu130
```

Every other dependency is pulled in from the `pyproject.toml` files. NURBSDiff is vendored as `cadmap._vendor.nurbsdiff`
(upstream ebd75a6 plus the `range_u`/`range_v` and `find_span_torch` patches),
and PyTorch3D is no longer required.

## Direct fitting

Provide targets as a list of (N_i, 3) point arrays, one segmented patch per face,
in the same order as solid.descendent:

```python
from cadmap.conversion.occ import step_to_solid, solid_to_device, create_diffsurf_of_solid
from cadmap.conversion.export import write_step
from cadmap.deformation.fitting import CADMapOptimizer

solid = step_to_solid("part.step")
solid_to_device(solid, "cpu")
create_diffsurf_of_solid(solid)
# Load your per-face segmented point arrays into targets.
optimizer = CADMapOptimizer(solid, targets=targets, device="cpu", n_epochs=500)
optimizer.run()
write_step(optimizer.export_occ_solid(), "fitted.step")
```

## MeshMap / processtrak initialization

After MeshMap produces each specimen's bound displacement field, add this step
to the experiment PRX:

```xml
<prx:step name="cadinitialize">
  <prx:elementmatch>dc:measurement/dc:submeas</prx:elementmatch>
  <prx:script name="CADMap_initialize.py"/>
</prx:step>
```

Put parameters in the experiment XLG. The parent measurement supplies
`dc:CADfilename` and the generated `dc:modal_cad_mapping`; each specimen supplies
`dc:measident`, `dc:dest` and the generated `dc:cad_bound_fit`. Available parameters
are `dc:cad_output_mode`, `dc:cad_continuity_samples`, `dc:cad_smoothing_weight`
and `dc:cad_ridge`.

`target_coord` exports in the scan frame, including the recorded rigid translation
and rotation. `original_coord` retains the original CAD coordinate frame. Outputs
are STEP, NPZ control points/UV data and a JSON validation report. Hashes, mesh
order and coordinate frames are checked. Invalid, open or self-intersecting
B-Reps fail initialization. Initialization approximates the fitted mesh field;
it is not a further point-cloud fit.

## Attribution

Liu, Lijie; Shen, Weijun; Zhang, Zhan; Holland, Stephen D.; Li, Qing; and
Krishnamurthy, Adarsh. Reconstruction of deformed trimmed NURBS solid models
using a reference CAD topology (2024).

Package metadata declares the MIT license. CADMap uses NURBSDiff for
differentiable NURBS evaluation and pythonocc-core/OpenCascade for B-Rep and STEP I/O.
