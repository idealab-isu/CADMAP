from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from cadmap.geometry.nurbs import BsplineSurface, Curve2d, Curve3d


class Topology:
    """Node in the B-Rep topology tree (solid → face → wire → edge → vertex)."""

    def __init__(self, topods_shape: Any) -> None:
        """
        Parameters
        ----------
        topods_shape:
            Any OCC ``TopoDS_*`` shape object.
        """
        self.ancestor: List["Topology"] = []
        self.descendent: List["Topology"] = []
        self.TopoDS = topods_shape
        self.geom: Optional[Any] = None   # 3-D geometry (BsplineSurface / Curve3d)
        self.geom2d: Optional[Any] = None  # 2-D geometry (Curve2d)

    def add_descendent(self, child: "Topology") -> None:
        self.descendent.append(child)

    def add_ancestor(self, parent: "Topology") -> None:
        self.ancestor.append(parent)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Topology):
            return NotImplemented
        return bool(self.TopoDS.IsSame(other.TopoDS))

    def __hash__(self) -> int:
        return hash(self.TopoDS)
