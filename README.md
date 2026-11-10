# CADMap 0.2.0

CADMap fits trimmed NURBS CAD models to segmented point clouds and initializes
CAD control points from MeshMap displacement fields through processtrak.

This copy retains the complete current Python package and original Git history.
Large fixtures, paper assets, caches and previous calculation outputs are
omitted from this checkout. Keep experiment data and
results outside this repository, for example in the sibling ../experiment folder.

## Installation

CADMap 0.2.0 is paired with MeshMap (`EigenFit`) 0.2.0. See the included
[INSTALL.md](INSTALL.md) for prerequisites, wheel installation and development setup.
With the three source directories available as siblings:

```powershell
conda create -n mapping312 -c conda-forge python=3.12 pythonocc-core h5py -y
conda activate mapping312
pip install -e ./limatix -e ./MeshMap -e ./CADMap
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

## Automatic UV reparameterization (processtrak)

Thin trims that fill a small part of their surface's parameter domain waste
control points. Run this step first after `copyinput`, before `buildmodel`:

```xml
<prx:step name="uvreparam">
  <prx:elementmatch>dc:measurement</prx:elementmatch>
  <prx:script name="CADMap_uv_reparameterize.py"/>
</prx:step>
```

It writes `<measident>_uv_reparameterized.step` and `.json` to `dc:dest` and
replaces `dc:CADfilename`, so COMSOL, MeshMap's CAD correspondence and
`cadinitialize` all use the same file. The input is kept as
`dc:CADfilename_original` and is always the source on re-runs.

Only face charts change; the solid's geometry and topology do not. A single-span
B-spline face is segmented to its padded trim enclosure when that at least halves
its domain (`restrict`). A plane or single-span B-spline face whose trim fills
less than `dc:uv_trigger_occupancy` (default 0.2) of that enclosure gets the
exact buffered polynomial strip chart (`strip`). Other surface types are kept.

Every changed face is checked on the final OpenCascade geometry: surface
position against the original (`dc:uv_geometry_tolerance_mm`, default 1e-7),
normal orientation, a regular Jacobian, pcurves inside the new domain, and
pcurve-on-surface against the 3D edge (`dc:uv_edge_tolerance_mm`, default 1e-6).
A face that fails keeps its original chart. The solid must then keep its
validity, topology counts, volume and area, also after STEP readback. If not,
the step copies the input unchanged (`dc:uv_reparam_on_failure` = `fallback`,
the default) or fails (`error`). `dc:uv_reparam` = `off` disables the step,
`dc:uv_buffer_fraction` (default 0.1) sets the strip buffer, and
`dc:uv_restrict_bspline` = `false` disables segmentation. The status is
recorded in `dc:uv_reparameterization_status`.

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

## Release scope and limitations

This release packages the existing main implementation, including automatic UV
reparameterization and the schema-1 MeshMap initialization interface. It does
not include the separate CAD initialization fix branches or the experimental
global-field initializer. Recorded C17 runs of this baseline with ridge=1 failed
the Boolean validity check; this release does not claim acceptance of that C17
workflow. Initialization requires checking the validation report for each input.
See [CHANGELOG.md](CHANGELOG.md) and [RELEASING.md](RELEASING.md).

## Attribution

Liu, Lijie; Shen, Weijun; Zhang, Zhan; Holland, Stephen D.; Li, Qing; and
Krishnamurthy, Adarsh. Reconstruction of deformed trimmed NURBS solid models
using a reference CAD topology (2024).

CADMap is distributed under the [MIT license](LICENSE). The vendored NURBSDiff
code retains its [BSD-3-Clause license](src/cadmap/_vendor/nurbsdiff/LICENSE).
CADMap uses NURBSDiff for
differentiable NURBS evaluation and pythonocc-core/OpenCascade for B-Rep and STEP I/O.
