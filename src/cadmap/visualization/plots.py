"""Visualisation utilities: matplotlib plots, Open3D scenes, OCC display, STEP export."""

from __future__ import annotations

import copy
import time
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import torch

from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
from OCC.Core.BRepLib import breplib, breplib_BuildCurves3d
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepTools import breptools
from OCC.Core.Geom import Geom_BSplineSurface
from OCC.Core.Geom2d import Geom2d_BSplineCurve
from OCC.Core.gp import gp_Pnt, gp_Pnt2d
from OCC.Core.TColgp import TColgp_Array1OfPnt, TColgp_Array1OfPnt2d, TColgp_Array2OfPnt
from OCC.Core.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal, TColStd_Array2OfReal
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_VERTEX
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import TopoDS_Shell, TopoDS_Solid, TopoDS_Shape, topods

from cadmap._internal.math_ops import batch_wn
from cadmap.geometry.topology import Topology


# ---------------------------------------------------------------------------
# 2-D / 3-D matplotlib helpers
# ---------------------------------------------------------------------------

def set_axes_equal(ax) -> None:
    """Force equal 3-D axis scaling so spheres look like spheres."""
    xl, yl, zl = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    r = 0.5 * max(abs(xl[1] - xl[0]), abs(yl[1] - yl[0]), abs(zl[1] - zl[0]))
    for lim, setter in zip([xl, yl, zl], [ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
        m = np.mean(lim)
        setter([m - r, m + r])


def set_camera(ax, fig, **camera_kwargs):
    for key, value in camera_kwargs.items():
        setattr(ax, key, value)
    return ax, fig


def plot_wire2d(wire2d, fig=None, ax=None):
    if fig is None:
        fig, ax = plt.subplots()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(wire2d.descendent)))
    for ci, edge in enumerate(wire2d.descendent):
        pts = np.array(edge.geom2d.evaluate_list(np.arange(0, 1, 0.01)))
        ax.plot(pts[:, 0], pts[:, 1], c=colors[ci])
        ax.plot(pts[0, 0], pts[0, 1], "o", c=colors[ci])
        ax.plot(pts[-1, 0], pts[-1, 1], "x", c=colors[ci])
    plt.show()
    return fig, ax


def plot_wire3d(wire2d, surf_ctp=None, fig=None, ax=None, color=None):
    surface = wire2d.ancestor[0].geom
    if surf_ctp is None:
        surf_ctp = surface.init_ctp
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
    colors = plt.cm.rainbow(np.linspace(0, 1, len(wire2d.descendent))) if color is None else [color] * len(wire2d.descendent)
    for ci, edge in enumerate(wire2d.descendent):
        uv = np.clip(np.array(edge.geom2d.evaluate_list(np.arange(0, 1, 0.01))), 0, 1)
        pts = np.array(surface.evaluate_list(surf_ctp, uv))
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], c=colors[ci])
        ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], marker="o", color=colors[ci])
        ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], marker="x", color=colors[ci])
    plt.show()
    return fig, ax


def plot_solid_wire(solid: Topology, ctp=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.rainbow(np.linspace(0, 1, len(solid.descendent)))
    for ci, face in enumerate(solid.descendent):
        surface = face.geom
        surf_ctp = surface.init_ctp if ctp is None else ctp[surface.ctp_index[0]: surface.ctp_index[1], :]
        for wire in face.descendent:
            fig, ax = plot_wire3d(wire, surf_ctp, fig, ax, color=colors[ci])
    plt.show()
    return fig, ax


def plot_surface(surface, ctp=None, fig=None, ax=None, delta=None, **plot_kwargs):
    fig = fig or plt.figure()
    ax = ax or fig.add_subplot(111, projection="3d")
    if delta is None:
        pts = surface.evalpts(surface.init_ctp if ctp is None else ctp)
        pts = pts.reshape(surface.uEvalPtSize, surface.vEvalPtSize, 3)
        mask = np.repeat(surface.inside_evalpts[..., np.newaxis], 3, -1)
        pts[~mask] = None
    else:
        gsurf = surface.create_3dNURBS(surface.init_ctp if ctp is None else ctp)
        u_v = np.arange(0, 1 + delta, delta)
        grid = np.array([(u, v) for u in u_v for v in u_v])
        pts = np.array(gsurf.evaluate_list(grid)).reshape(len(u_v), len(u_v), 3)
        uv_g = grid.reshape(len(u_v), len(u_v), 2)
        intrim = batch_wn(uv_g, surface.topology)
        pts[~np.repeat(intrim[..., np.newaxis], 3, -1)] = None
    ax.plot_surface(pts[:, :, 0], pts[:, :, 1], pts[:, :, 2], **plot_kwargs)
    return fig, ax


def plot_solid(solid: Topology, ctpall=None, fig=None, ax=None, delta=None, **plot_kwargs):
    fig = fig or plt.figure()
    ax = ax or fig.add_subplot(111, projection="3d")
    colors = plot_kwargs.pop("color", plt.cm.rainbow(np.linspace(0, 1, len(solid.descendent))))
    for fi, face in enumerate(solid.descendent):
        ctp = None if ctpall is None else ctpall[face.geom.ctp_index[0]: face.geom.ctp_index[1], :]
        plot_surface(face.geom, ctp=ctp, fig=fig, ax=ax, delta=delta, color=colors[fi], **plot_kwargs)
    return fig, ax


def plot_scatter_surface(surface, ctp=None, fig=None, ax=None, delta=None, color=None):
    fig = fig or plt.figure()
    ax = ax or fig.add_subplot(111, projection="3d")
    if delta is None:
        pts = surface.evalpts(
            surface.init_ctp if ctp is None else ctp.reshape(surface.size_u, surface.size_v, 3)
        ).reshape(surface.uEvalPtSize, surface.vEvalPtSize, 3)
        mask = np.repeat(surface.inside_evalpts[..., np.newaxis], 3, -1)
        pts[~mask] = None
    else:
        gsurf = surface.create_3dNURBS(
            surface.init_ctp if ctp is None else ctp.reshape(surface.size_u, surface.size_v, 3)
        )
        u_v = np.arange(0, 1 + delta, delta)
        grid = np.array([(u, v) for u in u_v for v in u_v])
        pts = np.array(gsurf.evaluate_list(grid)).reshape(len(u_v), len(u_v), 3)
        uv_g = grid.reshape(len(u_v), len(u_v), 2)
        intrim = batch_wn(uv_g, surface.topology)
        pts[~np.repeat(intrim[..., np.newaxis], 3, -1)] = None
    ax.scatter(pts[:, :, 0], pts[:, :, 1], pts[:, :, 2], color=color)
    return fig, ax


def plot_scatter_solid(solid: Topology, ctpall=None, fig=None, ax=None, delta=None):
    fig = fig or plt.figure()
    ax = ax or fig.add_subplot(111, projection="3d")
    colors = plt.cm.rainbow(np.linspace(0, 1, len(solid.descendent)))
    for fi, face in enumerate(solid.descendent):
        ctp = None if ctpall is None else ctpall[face.geom.ctp_index[0]: face.geom.ctp_index[1], :]
        plot_scatter_surface(face.geom, ctp=ctp, fig=fig, ax=ax, delta=delta, color=colors[fi])
    return fig, ax


def plot_target_points(targets: List[np.ndarray], color=None, ax=None, fig=None, s: int = 1):
    color = color or plt.cm.rainbow(np.linspace(0, 1, len(targets)))
    fig = fig or plt.figure()
    ax = ax or fig.add_subplot(111, projection="3d")
    for ci, pts in enumerate(targets):
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=color[ci], s=s)
    return ax, fig


# ---------------------------------------------------------------------------
# OCC shape → Open3D mesh
# ---------------------------------------------------------------------------

def occ_shape_to_open3d_mesh(shape, linear_deflection: float = 0.1, angular_deflection: float = 0.5) -> List:
    mesh_builder = BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
    mesh_builder.Perform()

    o3d_meshes = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = topods.Face(exp.Current())
        loc = face.Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri is None:
            exp.Next()
            continue
        trsf = loc.Transformation()
        verts = []
        for i in range(1, tri.NbNodes() + 1):
            p = tri.Node(i)
            p.Transform(trsf)
            verts.append([p.X(), p.Y(), p.Z()])
        tris = []
        for i in range(1, tri.NbTriangles() + 1):
            t = tri.Triangle(i)
            tris.append([t.Value(1) - 1, t.Value(2) - 1, t.Value(3) - 1])
        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(np.array(verts))
        m.triangles = o3d.utility.Vector3iVector(np.array(tris))
        m.compute_vertex_normals()
        o3d_meshes.append(m)
        exp.Next()
    return o3d_meshes


def plot_occ_shape(occ_shape: TopoDS_Shape) -> None:
    from OCC.Display.SimpleGui import init_display
    display, start_display, _, _ = init_display()
    display.DisplayShape(occ_shape, update=True)
    display.FitAll()
    start_display()


# ---------------------------------------------------------------------------
# OCC solid reconstruction and I/O
# ---------------------------------------------------------------------------

def convert_to_OCC_solid(solid: Topology, ctpall_new=None):
    """Rebuild an OCC ``TopoDS_Solid`` from the cadmap topology and (optionally) new control points."""
    builder = BRep_Builder()
    occ_solid = TopoDS_Solid()
    builder.MakeSolid(occ_solid)
    occ_shell = TopoDS_Shell()
    builder.MakeShell(occ_shell)
    occ_faces = []

    for face in solid.descendent:
        surf = face.geom
        poles = TColgp_Array2OfPnt(1, surf.size_u, 1, surf.size_v)
        for i in range(surf.size_u):
            for j in range(surf.size_v):
                if ctpall_new is None:
                    ctp_ij = surf.init_ctp[i, j]
                else:
                    idx = np.arange(surf.ctp_index[0], surf.ctp_index[1])
                    ctp_r = ctpall_new[idx[0]: idx[-1] + 1].reshape(surf.size_u, surf.size_v, 3)
                    if isinstance(ctp_r, torch.Tensor):
                        ctp_ij = ctp_r[i, j].detach().cpu().numpy()
                    else:
                        ctp_ij = ctp_r[i, j]
                poles.SetValue(i + 1, j + 1, gp_Pnt(float(ctp_ij[0]), float(ctp_ij[1]), float(ctp_ij[2])))

        ku, mu = np.unique(surf.knotvector_u, return_counts=True)
        kv, mv = np.unique(surf.knotvector_v, return_counts=True)

        def _arr1r(vals):
            a = TColStd_Array1OfReal(1, len(vals))
            for i, v in enumerate(vals): a.SetValue(i + 1, float(v))
            return a

        def _arr1i(vals):
            a = TColStd_Array1OfInteger(1, len(vals))
            for i, v in enumerate(vals): a.SetValue(i + 1, int(v))
            return a

        def _arr2r(mat):
            a = TColStd_Array2OfReal(1, mat.shape[0], 1, mat.shape[1])
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    a.SetValue(i + 1, j + 1, float(mat[i, j]))
            return a

        occ_surf = Geom_BSplineSurface(
            poles, _arr2r(surf.weight), _arr1r(ku), _arr1r(kv),
            _arr1i(mu), _arr1i(mv), surf.degree_u, surf.degree_v, False, False,
        )

        occ_wires_CCW, occ_wires_CW = [], []
        for wire in face.descendent:
            wm = BRepBuilderAPI_MakeWire()
            for edge_cls in wire.descendent:
                e2d = edge_cls.geom2d
                pts2d = TColgp_Array1OfPnt2d(1, e2d.size)
                for i in range(e2d.size):
                    pts2d.SetValue(i + 1, gp_Pnt2d(float(e2d.fix_ctp[i, 0]), float(e2d.fix_ctp[i, 1])))
                ku_e, mu_e = np.unique(e2d.knotvector, return_counts=True)
                w_e = TColStd_Array1OfReal(1, e2d.size)
                for i in range(e2d.size): w_e.SetValue(i + 1, float(e2d.weight[i]))
                trim2d = Geom2d_BSplineCurve(pts2d, w_e, _arr1r(ku_e), _arr1i(mu_e), e2d.degree, False)
                occ_edge = BRepBuilderAPI_MakeEdge(trim2d, occ_surf).Edge()
                breplib.BuildCurves3d(occ_edge)
                wm.Add(occ_edge)
            occ_wire = wm.Wire()
            if breptools.OuterWire(face.TopoDS).IsSame(wire.TopoDS):
                occ_wires_CCW.append(occ_wire)
            else:
                occ_wires_CW.append(occ_wire)

        if not occ_wires_CCW:
            fm = BRepBuilderAPI_MakeFace(occ_surf, 1e-7)
        else:
            fm = BRepBuilderAPI_MakeFace(occ_surf, occ_wires_CCW[0], True)
        for w in occ_wires_CW:
            fm.Add(w)
        occ_face = fm.Face()
        occ_faces.append(occ_face)
        builder.Add(occ_shell, occ_face)

    builder.Add(occ_solid, occ_shell)
    return occ_solid, None, occ_faces


def write_step(occ_shape: TopoDS_Shape, filename: str) -> None:
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.Interface import Interface_Static_SetCVal
    from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer

    writer = STEPControl_Writer()
    Interface_Static_SetCVal("write.step.schema", "AP214")
    writer.Transfer(occ_shape, STEPControl_AsIs)
    if writer.Write(filename) != IFSelect_RetDone:
        raise RuntimeError(f"Failed to write STEP file: {filename}")


# ---------------------------------------------------------------------------
# Open3D point-cloud helpers
# ---------------------------------------------------------------------------

def convert_to_OCC_ptc(ptc_array: np.ndarray, colors=None) -> o3d.geometry.PointCloud:
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(ptc_array)
    if colors is not None:
        pc.colors = o3d.utility.Vector3dVector(np.tile(colors, (len(ptc_array), 1)))
    return pc


def convert_to_OCC_line(ptc_array: np.ndarray, solid: Topology, colors=None) -> List:
    line_sets = []
    for face in solid.descendent:
        surf = face.geom
        ctp_r = ptc_array[surf.ctp_index[0]: surf.ctp_index[1]].reshape(surf.size_u, surf.size_v, 3)
        lines = []
        for i in range(surf.size_u):
            for j in range(surf.size_v - 1):
                lines.append([i * surf.size_v + j, i * surf.size_v + j + 1])
        for i in range(surf.size_u - 1):
            for j in range(surf.size_v):
                lines.append([i * surf.size_v + j, (i + 1) * surf.size_v + j])
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(ctp_r.reshape(-1, 3))
        ls.lines = o3d.utility.Vector2iVector(lines)
        line_sets.append(ls)
    return line_sets


# ---------------------------------------------------------------------------
# Interactive Open3D animation
# ---------------------------------------------------------------------------

def o3d_animation(solid: Topology, saved_inpCtrlPts_all, ptc_list, iter_frames: int = 1) -> None:
    from tqdm import tqdm

    if iter_frames != 1:
        frames = [
            torch.lerp(saved_inpCtrlPts_all[i], saved_inpCtrlPts_all[i + 1], j / iter_frames)
            for i in range(len(saved_inpCtrlPts_all) - 1)
            for j in range(iter_frames)
        ] + [saved_inpCtrlPts_all[-1]]
    else:
        frames = saved_inpCtrlPts_all

    OCC_ctp_list = [convert_to_OCC_ptc(f.numpy(), np.array([1, 0, 0], dtype=float)) for f in frames]
    OCC_ctpmesh_list = [convert_to_OCC_line(f.numpy(), solid) for f in frames]
    OCC_ptc_list = [convert_to_OCC_ptc(p) for p in ptc_list]

    mesh_states = []
    for frame in tqdm(frames, desc="Rendering frames"):
        occ_solid, _, _ = convert_to_OCC_solid(solid, frame.detach().cpu())
        mesh_states.append(occ_shape_to_open3d_mesh(occ_solid))

    gui.Application.instance.initialize()
    app = VisualizationApp(
        mesh_states,
        OCC_ptc_list=OCC_ptc_list,
        OCC_ctp_list=OCC_ctp_list,
        OCC_ctpmesh_list=OCC_ctpmesh_list,
    )
    gui.Application.instance.run()


# ---------------------------------------------------------------------------
# VisualizationApp and App classes
# ---------------------------------------------------------------------------

class VisualizationApp:
    """Open3D GUI application for frame-by-frame animation of CAD deformation."""

    MENU_QUIT = 1
    CTP_SHOW = 2
    PTC_SHOW = 3
    FRAME_TEXT_SHOW = 4
    CAMERA_TEXT_SHOW = 5

    def __init__(self, mesh_states, **kwargs):
        self.mesh_states = mesh_states
        self.steps = len(mesh_states)
        self.current_step = 0
        self.OCC_ptc_list = kwargs.get("OCC_ptc_list", [])
        self.OCC_ctp_list = kwargs.get("OCC_ctp_list", [])
        self.OCC_ctpmesh_list = kwargs.get("OCC_ctpmesh_list", [])
        self.opacity = kwargs.get("opacity", 0.85)
        self.show_ctp = True
        self.show_ptc = True
        self.show_frame_text = True
        self.show_camera_text = True
        self.last_key_time = 0.0
        self.MIN_INTERVAL = 0.15
        self.angle_id = None

        n = len(self.OCC_ptc_list)
        cmap = plt.get_cmap("brg", max(n, 1))
        self.colors = [list(cmap(i))[:3] + [self.opacity] for i in range(n)]

        self.window = gui.Application.instance.create_window("Animation", 1024, 768, 0, 30)
        menu = gui.Menu()
        m = gui.Menu()
        m.add_item("Quit", self.MENU_QUIT)
        m.add_item("Show/Hide CPs", self.CTP_SHOW)
        m.add_item("Show/Hide Point Cloud", self.PTC_SHOW)
        menu.add_menu("menu", m)
        gui.Application.instance.menubar = menu
        self.window.set_on_menu_item_activated(self.MENU_QUIT, self._quit)
        self.window.set_on_menu_item_activated(self.CTP_SHOW, self._toggle_ctp)
        self.window.set_on_menu_item_activated(self.PTC_SHOW, self._toggle_ptc)

        self.widget = gui.SceneWidget()
        self.widget.scene = rendering.Open3DScene(self.window.renderer)
        self.widget.scene.camera.look_at([0, 0, 0], [100, 100, 100], [0, 0, 1])
        self.widget.scene.set_background([1, 1, 1, 1])
        self.label_id = self.widget.add_3d_label(np.array([0, 0, 0]), f"Frame: {self.current_step}")
        self.window.add_child(self.widget)
        self.widget.set_on_mouse(self._on_mouse)
        self.window.set_on_key(self._on_key)
        self.update_scene()

    def _quit(self): gui.Application.instance.quit()
    def _toggle_ctp(self): self.show_ctp = not self.show_ctp; self.update_scene()
    def _toggle_ptc(self): self.show_ptc = not self.show_ptc; self.update_scene()

    def _on_mouse(self, event):
        if event.type == gui.MouseEvent.MOVE and self.show_camera_text:
            m = self.widget.scene.camera.get_model_matrix()
            info = f"Pos: {m[:3,3]}\nFront: {m[:3,2]}\nUp: {m[:3,1]}"
            if self.angle_id is not None:
                self.widget.remove_3d_label(self.angle_id)
            self.angle_id = self.widget.add_3d_label(np.array([-2, 10, 0]), info)
        return gui.Widget.EventCallbackResult.HANDLED

    def update_scene(self):
        self.widget.scene.clear_geometry()
        meshes = self.mesh_states[self.current_step]
        if not isinstance(meshes, list):
            meshes = [meshes]
        for i, m in enumerate(meshes):
            mat = rendering.MaterialRecord()
            mat.shader = "defaultLitTransparency"
            mat.base_color = self.colors[i] if i < len(self.colors) else [0.8, 0.8, 0.8, self.opacity]
            m.compute_vertex_normals()
            self.widget.scene.add_geometry(f"mesh_{i}", m, mat)

        if self.show_ctp and self.OCC_ctp_list:
            cm = rendering.MaterialRecord(); cm.shader = "defaultUnlit"; cm.point_size = 5.0
            self.widget.scene.add_geometry("ctp", self.OCC_ctp_list[self.current_step], cm)
            lm = rendering.MaterialRecord(); lm.shader = "unlitLine"; lm.line_width = 2.0; lm.base_color = [0, 0, 1, 1]
            for i, ls in enumerate(self.OCC_ctpmesh_list[self.current_step]):
                self.widget.scene.add_geometry(f"ctpline_{i}", ls, lm)

        if self.show_ptc:
            for i, ptc in enumerate(self.OCC_ptc_list):
                pm = rendering.MaterialRecord(); pm.shader = "defaultUnlit"
                pm.base_color = self.colors[i] if i < len(self.colors) else [0.5, 0.5, 0.5, 1]
                pm.point_size = 5.0
                self.widget.scene.add_geometry(f"ptc_{i}", ptc, pm)

        if self.show_frame_text:
            if hasattr(self, "label_id"):
                self.widget.remove_3d_label(self.label_id)
            self.label_id = self.widget.add_3d_label(np.array([0, 0, 0]), f"Frame: {self.current_step}")

    def _on_key(self, event):
        now = time.perf_counter()
        if now - self.last_key_time < self.MIN_INTERVAL:
            return False
        self.last_key_time = now
        if event.key == gui.KeyName.RIGHT:
            self.current_step = min(self.current_step + 1, self.steps - 1)
            self.update_scene()
        elif event.key == gui.KeyName.LEFT:
            self.current_step = max(self.current_step - 1, 0)
            self.update_scene()
        return True


class App:
    """Lightweight Open3D GUI for displaying static/animated geometry lists."""

    MENU_QUIT = 1

    def __init__(self, geom_list_list, geom_obj_list, **kwargs):
        self.geom_list_list = geom_list_list
        self.steps = len(geom_list_list[0]) if geom_list_list else 1
        self.current_step = 0
        self.geom_obj_list = geom_obj_list
        self.last_key_time = 0.0
        self.MIN_INTERVAL = 0.15

        self.window = gui.Application.instance.create_window("Viewer", 1024, 768, 0, 30)
        if gui.Application.instance.menubar is None:
            m = gui.Menu(); m2 = gui.Menu(); m2.add_item("Quit", self.MENU_QUIT)
            m.add_menu("menu", m2); gui.Application.instance.menubar = m
        self.window.set_on_menu_item_activated(self.MENU_QUIT, lambda: gui.Application.instance.quit())

        self.widget = gui.SceneWidget()
        self.widget.scene = rendering.Open3DScene(self.window.renderer)
        self.widget.scene.camera.look_at([0, 0, 0], [100, 100, 100], [0, 0, 1])
        self.widget.scene.set_background([1, 1, 1, 1])
        self.label_id = self.widget.add_3d_label(np.array([0, 0, 0]), f"Frame: {self.current_step}")
        self.window.add_child(self.widget)
        self.window.set_on_key(self._on_key)
        self.update_scene()

    def _add_geom(self, geom):
        mat = rendering.MaterialRecord()
        if isinstance(geom, o3d.geometry.PointCloud):
            mat.shader = "defaultUnlit"; mat.point_size = 3.0
        else:
            mat.shader = "defaultLit"; geom.compute_vertex_normals()
        self.widget.scene.add_geometry(str(id(geom)), geom, mat)

    def update_scene(self):
        self.widget.scene.clear_geometry()
        for g in (self.geom_obj_list or []):
            self._add_geom(g)
        for gl in (self.geom_list_list or []):
            self._add_geom(gl[self.current_step])
        self.label_id = self.widget.add_3d_label(np.array([0, 0, 0]), f"Frame: {self.current_step}")

    def _on_key(self, event):
        now = time.perf_counter()
        if now - self.last_key_time < self.MIN_INTERVAL:
            return False
        self.last_key_time = now
        if event.key == gui.KeyName.RIGHT:
            self.current_step = min(self.current_step + 1, self.steps - 1)
            self.update_scene()
        elif event.key == gui.KeyName.LEFT:
            self.current_step = max(self.current_step - 1, 0)
            self.update_scene()
        return True
