# Install CADMap 0.2.0 and MeshMap / EigenFit 0.2.0

The supported validation environment for this release is Windows with Python
3.12. Package metadata retains Python >=3.9; other versions have not been
validated in this release. CADMap depends on EigenFit; EigenFit uses limatix
for processtrak. Install external prerequisites before installing the wheels.

## Prerequisites

```powershell
conda create -n mapping-release -c conda-forge python=3.12 pythonocc-core h5py -y
conda activate mapping-release
```

Keep h5py on the conda HDF5 stack used by pythonocc-core/VTK. Install PyTorch
for your platform and CPU/GPU from the official PyTorch installation selector
at https://pytorch.org/get-started/locally/. A GPU is optional for package
installation and the synthetic checks. MeshMap pins open3d below 0.20 because
the existing interactive picker relies on depth-buffer behavior.

## limatix dependency

Supply and install the limatix source approved for your processtrak environment:

```powershell
python -m pip install /path/to/approved/limatix
```

The local validation uses limatix commit
`23bd402deb078945fc55f978e828de18c8fd5ed5`. It contains local dependency and
processtrak compatibility changes that are not in the upstream GitHub branches
at release preparation time. These changes are not published by this release;
an arbitrary upstream limatix checkout has not been validated as a replacement.
Obtain that approved snapshot before attempting to reproduce the full workflow.

## Install the release wheels

Download both 0.2.0 wheels from the repositories' GitHub Releases into one folder:

```powershell
python -m pip install ./eigenfit-0.2.0-py3-none-any.whl ./cadmap-0.2.0-py3-none-any.whl
python -m pip check
python -c "import h5py, EigenFit, cadmap; print(EigenFit.__version__, cadmap.__version__)"
meshmap --help
eigenfit-resume --help
```

Both versions should print `0.2.0`. Add the `ansys` extra when installing
EigenFit if you need the ANSYS Python clients. COMSOL execution requires COMSOL,
MATLAB and LiveLink for MATLAB; ANSYS execution requires its own installation
and licenses. Those programs are not bundled into either wheel.

## Development from source

Use these commands from a parent directory containing the three source folders:

```powershell
python -m pip install -e ./limatix -e ./MeshMap -e ./CADMap
```

Select the v0.2.0 tags to reproduce this release. Working on a fix branch changes
the runtime behavior of an editable installation. Keep experiment inputs and
outputs outside the source repositories and pass their paths explicitly.

## Build from source

```powershell
python -m pip install build twine
python -m build
python -m twine check dist/*
```

Run inside the selected repository. Prefer the supplied sdist over GitHub's
automatic source ZIP when building without Git: EigenFit's sdist contains the
version metadata needed to rebuild its wheel without Git history.
