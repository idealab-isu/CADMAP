# Installation

## Standard Install

```bash
git clone https://github.com/yourusername/cadmap.git
cd cadmap
pip install -e ".[dev]"
```

## Manual Dependencies

The following packages cannot be installed via PyPI and must be set up separately:

### pythonocc-core

Install via conda:

```bash
conda install -c conda-forge pythonocc-core
```

### NURBSDiff

```bash
pip install git+https://github.com/idealab-isu/NURBSDiff.git
```

### pytorch3d

Follow the [official pytorch3d install guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for your CUDA version.

## Verifying

```python
import cadmap
print(cadmap.__version__)
```
