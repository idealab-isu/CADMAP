# Changelog

## 0.2.0 - 2026-11-09

- Package the existing main baseline (`f2ed7c6`) for distribution.
- Declare runtime dependencies, vendor NURBSDiff with its license, and remove
  the PyTorch3D requirement.
- Provide automatic, geometry-validated UV reparameterization and accept
  `CADfilename_original` without causing a later processtrak mergeinput conflict.
- Read MeshMap per-face segmented point clouds and retain schema-1 bound-field
  initialization with coordinate-frame and source-hash checks.
- Use a single version source, include installation/release instructions and
  license files, and pair with EigenFit >=0.2.0,<0.3.

### Known limitations

The separate initialization fixes and global-field initializer are excluded.
Recorded C17 baseline runs with ridge=1 failed Boolean validation. Per-input
validation remains required; packaging checks do not establish full C17 or
licensed-solver acceptance. Large datasets and external solver installations
are not distributed with this package.
