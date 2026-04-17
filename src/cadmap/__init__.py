"""
CADMap: deforms a CAD model to a pre-segmented point cloud while preserving
CAD topology and water-tight properties.
"""

from cadmap.geometry.nurbs import BsplineSurface, Curve2d, Curve3d
from cadmap.geometry.topology import Topology
from cadmap.conversion.occ import SolidConvertor
from cadmap.sampling.pointcloud import sample_points_from_solid, extract_pc
from cadmap.deformation.fitting import CADMapOptimizer

__version__ = "0.1.0"
__all__ = [
    "BsplineSurface",
    "Curve2d",
    "Curve3d",
    "Topology",
    "SolidConvertor",
    "sample_points_from_solid",
    "extract_pc",
    "CADMapOptimizer",
]
