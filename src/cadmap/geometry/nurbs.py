from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import numpy as np
from geomdl import NURBS
from geomdl.construct import extract_curves
from geomdl.operations import length_curve
from NURBSDiff.surf_eval import SurfEval

import torch

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
from OCC.Core.BRepLib import breplib
from OCC.Core.BRepLProp import BRepLProp_CLProps
from OCC.Core.BRepTools import breptools
from OCC.Core.Geom import Geom_BSplineSurface
from OCC.Core.Geom2d import Geom2d_BSplineCurve
from OCC.Core.Geom2dAdaptor import Geom2dAdaptor_Curve
from OCC.Core.GCPnts import GCPnts_UniformAbscissa
from OCC.Core.gp import gp_Pnt, gp_Pnt2d
from OCC.Core.TColgp import TColgp_Array1OfPnt, TColgp_Array1OfPnt2d, TColgp_Array2OfPnt
from OCC.Core.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal, TColStd_Array2OfReal


class BsplineSurface:
    """NURBS surface geometry with differentiable evaluation support."""

    uvEvalPtSize_max: int = int(64 * 64 / 16)

    def __init__(
        self,
        init_ctp: Union[list, np.ndarray],
        knotvector_u: Union[list, np.ndarray],
        knotvector_v: Union[list, np.ndarray],
        surface_id: Union[str, int],
        ctp_start_index: int,
        periodic_u: bool,
        periodic_v: bool,
        degree_u: Optional[int] = None,
        degree_v: Optional[int] = None,
        size_u: Optional[int] = None,
        size_v: Optional[int] = None,
        weight: Optional[Union[list, np.ndarray]] = None,
        param_bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        self.init_ctp = np.array(init_ctp)
        if self.init_ctp.ndim < 3:
            if size_u is None or size_v is None:
                raise ValueError(
                    "Provide either a 3-D control-point array or a 2-D array "
                    "together with size_u and size_v."
                )
            self.init_ctp = self.init_ctp.reshape(size_u, size_v, 3)

        self.knotvector_u = np.array(knotvector_u)
        self.knotvector_v = np.array(knotvector_v)

        self.size_u: int = size_u if size_u is not None else self.init_ctp.shape[0]
        self.size_v: int = size_v if size_v is not None else self.init_ctp.shape[1]

        self.weight: np.ndarray = (
            np.array(weight).reshape(self.size_u, self.size_v)
            if weight is not None
            else np.ones((self.size_u, self.size_v))
        )

        self.degree_u: int = (
            degree_u if degree_u is not None
            else len(knotvector_u) - self.size_u - 1
        )
        self.degree_v: int = (
            degree_v if degree_v is not None
            else len(knotvector_v) - self.size_v - 1
        )

        self.surf_id: str = (
            surface_id if isinstance(surface_id, str) else f"Surface_{surface_id}"
        )
        self.ctp_index: Tuple[int, int] = (
            ctp_start_index,
            self.size_u * self.size_v + ctp_start_index,
        )
        self.periodic_u: bool = periodic_u
        self.periodic_v: bool = periodic_v

        # (u_min, u_max, v_min, v_max) of the surface this one was converted
        # from.  Knot vectors are normalised to [0, 1], so this is the affine
        # map needed to bring the original pcurves into our parameter space.
        self.param_bounds: Tuple[float, float, float, float] = (
            param_bounds if param_bounds is not None else (0.0, 1.0, 0.0, 1.0)
        )

        self.device: str = "cpu"
        self.DiffSurf: Optional[SurfEval] = None
        self.inside_evalpts: Optional[np.ndarray] = None
        self.topology = None  # back-ref to parent Topology node

        self.uEvalPtSize: int = 4
        self.vEvalPtSize: int = 4
        self.set_uv_EvalPts()

    # ------------------------------------------------------------------
    # OCC / geomdl conversions
    # ------------------------------------------------------------------

    def _to_geomsurface(self, ctp: Optional[np.ndarray] = None) -> Geom_BSplineSurface:
        """Return an OCC ``Geom_BSplineSurface`` from *ctp* (defaults to init_ctp)."""
        # float64 throughout: OCC's gp_Pnt rejects numpy float32.
        if ctp is None:
            ctp = self.init_ctp.astype(np.float64).reshape(self.size_u, self.size_v, 3)
        else:
            ctp = np.asarray(ctp, dtype=np.float64).reshape(self.size_u, self.size_v, 3)
        weight = self.weight.reshape(self.size_u, self.size_v)

        array_pnt = TColgp_Array2OfPnt(1, self.size_u, 1, self.size_v)
        array_weight = TColStd_Array2OfReal(1, self.size_u, 1, self.size_v)
        for i in range(self.size_u):
            for j in range(self.size_v):
                array_pnt.SetValue(i + 1, j + 1, gp_Pnt(*ctp[i, j, :]))
                array_weight.SetValue(i + 1, j + 1, float(weight[i, j]))

        uk, uc = np.unique(self.knotvector_u, return_counts=True)
        vk, vc = np.unique(self.knotvector_v, return_counts=True)

        def _make_knot_arrays(knots, mults):
            ka = TColStd_Array1OfReal(1, len(knots))
            ma = TColStd_Array1OfInteger(1, len(mults))
            for i, (k, m) in enumerate(zip(knots, mults)):
                ka.SetValue(i + 1, float(k))
                ma.SetValue(i + 1, int(m))
            return ka, ma

        ku_arr, mu_arr = _make_knot_arrays(uk, uc)
        kv_arr, mv_arr = _make_knot_arrays(vk, vc)

        return Geom_BSplineSurface(
            array_pnt, array_weight, ku_arr, kv_arr, mu_arr, mv_arr,
            self.degree_u, self.degree_v,
        )

    def _to_OCC_face(self):
        """Build an OCC ``TopoDS_Face`` from this surface and its trim wires."""
        from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeEdge
        from OCC.Core.BRepLib import breplib
        from OCC.Core.BRepTools import breptools

        occ_surface = self._to_geomsurface()
        face_topo = self.topology

        occ_wires_CCW = []
        occ_wires_CW = []
        for wire_topo in face_topo.descendent:
            wire_builder = BRepBuilderAPI_MakeWire()
            for edge_topo in wire_topo.descendent:
                occ_curve2d = edge_topo.geom2d._to_Geom2d_BSplineCurve()
                edge_maker = BRepBuilderAPI_MakeEdge(occ_curve2d, occ_surface)
                if edge_maker.IsDone():
                    occ_edge = edge_maker.Edge()
                    breplib.BuildCurves3d(occ_edge, 1e-6)
                    wire_builder.Add(occ_edge)
                else:
                    print(f"Warning: Failed to reconstruct edge {edge_topo}")

            if not wire_builder.IsDone():
                raise RuntimeError("Wire not done.")

            if breptools.OuterWire(face_topo.TopoDS).IsSame(wire_topo.TopoDS):
                occ_wires_CCW.append(wire_builder.Wire())
            else:
                occ_wires_CW.append(wire_builder.Wire())

        if len(occ_wires_CCW) == 0:
            face_builder = BRepBuilderAPI_MakeFace(occ_surface, 1e-6)
        elif len(occ_wires_CCW) == 1:
            face_builder = BRepBuilderAPI_MakeFace(occ_surface, occ_wires_CCW[0], True)
        else:
            raise ValueError("More than one CCW wire in the face.")
        for wire in occ_wires_CW:
            face_builder.Add(wire)

        if not face_builder.IsDone():
            raise RuntimeError("Face builder not done.")
        occ_face = face_builder.Face()
        if hasattr(face_topo, "TopoDS"):
            occ_face.Orientation(face_topo.TopoDS.Orientation())
        return occ_face

    def create_3dNURBS(self, ctp: np.ndarray) -> NURBS.Surface:
        """Return a geomdl NURBS surface from *ctp*."""
        if not isinstance(ctp, np.ndarray):
            ctp = np.array(ctp)
        surf3d = NURBS.Surface()
        surf3d.degree_u = self.degree_u
        surf3d.degree_v = self.degree_v
        ctp = ctp.reshape(self.size_u, self.size_v, 3)
        ctp = np.concatenate((ctp, self.weight.reshape(self.size_u, self.size_v, 1)), -1)
        ctp[..., :3] = ctp[..., :3] * ctp[..., 3:]
        surf3d.set_ctrlpts(ctp.reshape(-1, 4).tolist(), self.size_u, self.size_v)
        surf3d.knotvector_u = self.knotvector_u
        surf3d.knotvector_v = self.knotvector_v
        surf3d.delta = 0.01
        return surf3d

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _calculate_uv_EvalPts(self) -> Tuple[int, int]:
        nurbs3d = self.create_3dNURBS(self.init_ctp)
        curves_dict = extract_curves(nurbs3d)
        u_len = np.array([length_curve(c) for c in curves_dict["u"]]).mean()
        v_len = np.array([length_curve(c) for c in curves_dict["v"]]).mean()
        unit_u = np.sqrt(self.uvEvalPtSize_max / (v_len / u_len))
        u_sz = max(int(np.floor(unit_u)), 4)
        v_sz = max(int(np.floor(unit_u * (v_len / u_len))), 4)
        return (int(u_sz / 2) * 2, int(v_sz / 2) * 2)

    def set_uv_EvalPts(self, uvsize: Optional[Tuple[int, int]] = None) -> None:
        if uvsize is None:
            uvsize = self._calculate_uv_EvalPts()
        self.uEvalPtSize, self.vEvalPtSize = uvsize

    def get_UVBounds(self) -> Tuple[float, float, float, float]:
        occ_face = self._to_OCC_face()
        u_min, u_max, v_min, v_max = breptools.UVBounds(occ_face)
        return (
            max(0.0, u_min), min(1.0, u_max),
            max(0.0, v_min), min(1.0, v_max),
        )

    def create_diff_NURBS(self) -> None:
        """Instantiate the NURBSDiff ``SurfEval`` module on ``self.DiffSurf``."""
        u_min, u_max, v_min, v_max = self.get_UVBounds()
        for _ in range(6):
            torch.cuda.empty_cache()
            surf = SurfEval(
                self.size_u - 1, self.size_v - 1,
                knot_u=self.knotvector_u, knot_v=self.knotvector_v,
                dimension=3, p=self.degree_u, q=self.degree_v,
                out_dim_u=self.uEvalPtSize, out_dim_v=self.vEvalPtSize,
                range_u=(u_min, u_max), range_v=(v_min, v_max),
                device=self.device,
            )
            self.DiffSurf = surf
            tmp = self.evalpts(self.init_ctp)
            if not any(np.sum(np.isnan(x)) for x in tmp):
                return
        self.DiffSurf = None
        raise RuntimeError(
            "Failed to create differentiable surface — try decreasing uvEvalPtSize_max."
        )

    def evalpts(self, ctp: np.ndarray) -> np.ndarray:
        """Evaluate surface points via NURBSDiff (requires ``create_diff_NURBS`` first)."""
        return self.DiffSurf(
            torch.cat((
                torch.tensor(
                    ctp * self.weight[..., np.newaxis],
                    dtype=torch.float, device=self.device,
                ).unsqueeze(0),
                torch.tensor(
                    self.weight, dtype=torch.float, device=self.device,
                ).unsqueeze(0).unsqueeze(-1),
            ), axis=-1)
        ).cpu().detach().numpy()

    def evaluate_single(self, ctp: np.ndarray, param: Any) -> List[float]:
        return self.create_3dNURBS(ctp).evaluate_single(param)

    def evaluate_list(self, ctp: np.ndarray, param_list: Any) -> List[List[float]]:
        return self.create_3dNURBS(ctp).evaluate_list(param_list)

    def to_device(self, device: str) -> None:
        self.device = device

    def __str__(self) -> str:
        return "Surface"

    def __repr__(self) -> str:
        return self.surf_id


# ---------------------------------------------------------------------------


class Curve2d:
    """Immutable 2-D NURBS trim curve (parametric space)."""

    EvalPtSize_max: int = 512

    def __init__(
        self,
        init_ctp: Union[list, np.ndarray],
        knotvector: Union[list, np.ndarray],
        curv_id: Union[str, int],
        periodic: bool,
        degree: Optional[int] = None,
        size: Optional[int] = None,
        weight: Optional[Union[list, np.ndarray]] = None,
        orientation: int = 0,
    ) -> None:
        self.fix_ctp = np.array(init_ctp)
        if self.fix_ctp.ndim < 2:
            if size is None:
                raise ValueError(
                    "Provide either a 2-D control-point array or a 1-D array with size."
                )
            self.fix_ctp = self.fix_ctp.reshape(size, 2)

        self.curv_id: str = (
            curv_id if isinstance(curv_id, str) else f"Curve2d_{curv_id}"
        )
        self.size: int = size if size is not None else self.fix_ctp.shape[0]
        self.degree: int = (
            degree if degree is not None else len(knotvector) - self.size - 1
        )
        self.knotvector = np.array(knotvector)
        self.weight: np.ndarray = (
            np.array(weight).reshape(self.fix_ctp.shape[:-1])
            if weight is not None
            else np.ones(self.fix_ctp.shape[:-1])
        )
        self.periodic: bool = periodic
        self.orientation: int = orientation
        self.topology = None  # back-ref to parent Topology node

    # ------------------------------------------------------------------
    # OCC / geomdl conversions
    # ------------------------------------------------------------------

    def _to_Geom2d_BSplineCurve(self) -> Geom2d_BSplineCurve:
        array_pt = TColgp_Array1OfPnt2d(1, self.size)
        array_w = TColStd_Array1OfReal(1, self.size)
        for i in range(self.size):
            array_pt.SetValue(i + 1, gp_Pnt2d(float(self.fix_ctp[i, 0]), float(self.fix_ctp[i, 1])))
            array_w.SetValue(i + 1, float(self.weight[i]))

        uk, uc = np.unique(self.knotvector, return_counts=True)
        ka = TColStd_Array1OfReal(1, len(uk))
        ma = TColStd_Array1OfInteger(1, len(uc))
        for i, (k, m) in enumerate(zip(uk, uc)):
            ka.SetValue(i + 1, float(k))
            ma.SetValue(i + 1, int(m))

        return Geom2d_BSplineCurve(array_pt, array_w, ka, ma, self.degree, self.periodic)

    def _to_geomdl(self) -> NURBS.Curve:
        curve = NURBS.Curve()
        curve.degree = self.degree
        curve.set_ctrlpts(
            [[pt * w for pt in p] + [w] for p, w in zip(self.fix_ctp.tolist(), self.weight.tolist())]
        )
        curve.knotvector = self.knotvector
        return curve

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate2d_single(self, point: float) -> List[float]:
        return self._to_geomdl().evaluate_single(point)

    def evaluate_list(self, points: Any) -> List[List[float]]:
        return self._to_geomdl().evaluate_list(points)

    def UniformAbscissaSampling(self, num_points: int) -> List[List[float]]:
        bspline = self._to_Geom2d_BSplineCurve()
        adaptor = Geom2dAdaptor_Curve(bspline)
        algo = GCPnts_UniformAbscissa(adaptor, num_points, -1)
        if not algo.IsDone():
            return []
        return [[bspline.Value(algo.Parameter(i)).X(), bspline.Value(algo.Parameter(i)).Y()]
                for i in range(1, algo.NbPoints() + 1)]

    def __str__(self) -> str:
        return "Curve2d"

    def __repr__(self) -> str:
        return self.curv_id


# ---------------------------------------------------------------------------


class Curve3d:
    """Immutable 3-D NURBS edge curve."""

    EvalPtSize_max: int = 512

    def __init__(
        self,
        init_ctp: Union[list, np.ndarray],
        knotvector: Union[list, np.ndarray],
        curv_id: Union[str, int],
        periodic: bool,
        degree: Optional[int] = None,
        size: Optional[int] = None,
        weight: Optional[Union[list, np.ndarray]] = None,
    ) -> None:
        self.init_ctp = np.array(init_ctp)
        if self.init_ctp.ndim < 2:
            if size is None:
                raise ValueError(
                    "Provide either a 2-D control-point array or a 1-D array with size."
                )
            self.init_ctp = self.init_ctp.reshape(size, 3)

        self.curv_id: str = (
            curv_id if isinstance(curv_id, str) else f"Curve3d_{curv_id}"
        )
        self.size: int = size if size is not None else self.init_ctp.shape[0]
        self.degree: int = (
            degree if degree is not None else len(knotvector) - self.size - 1
        )
        self.knotvector = np.array(knotvector)
        self.weight: np.ndarray = (
            np.array(weight).reshape(self.init_ctp.shape[:-1])
            if weight is not None
            else np.ones(self.init_ctp.shape[:-1])
        )
        self.periodic: bool = periodic
        self.topology: List = []  # back-refs to parent Topology nodes (edge can be shared)

    # ------------------------------------------------------------------
    # geomdl conversion
    # ------------------------------------------------------------------

    def _to_geomdl(self, ctp: Union[list, np.ndarray]) -> NURBS.Curve:
        if not isinstance(ctp, np.ndarray):
            ctp = np.array(ctp)
        curve = NURBS.Curve()
        curve.degree = self.degree
        curve.set_ctrlpts(
            [[pt * w for pt in p] + [w] for p, w in zip(ctp.tolist(), self.weight.tolist())]
        )
        curve.knotvector = self.knotvector
        return curve

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate3d_single(self, point: float, ctp: Union[list, np.ndarray]) -> List[float]:
        return self._to_geomdl(ctp).evaluate_single(point)

    def evaluate_list(self, points: Any, ctp: Union[list, np.ndarray]) -> List[List[float]]:
        return self._to_geomdl(ctp).evaluate_list(points)

    def IsSame(self, other: object) -> bool:
        if other is None:
            return False
        if not isinstance(other, Curve3d):
            raise TypeError("Expected a Curve3d instance.")
        return self.curv_id == other.curv_id

    def __str__(self) -> str:
        return "Curve3d"

    def __repr__(self) -> str:
        return self.curv_id
