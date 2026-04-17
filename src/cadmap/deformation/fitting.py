"""CADMapOptimizer: topology-preserving NURBS deformation to a target point cloud."""

from __future__ import annotations

import copy
import functools
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from cadmap.deformation.constraints import (
    build_constraint_matrix,
    compute_local_basis,
    get_constraint_rows_for_edges,
    get_cp_indices_of_faces,
    get_edges_of_faces,
    get_unique_vertices,
    get_connectivity_of_vertex,
)
from cadmap.deformation.mesh_utils import create_customized_trimesh, find_closest_coord_on_mesh
from cadmap.geometry.topology import Topology


class CADMapOptimizer:
    """Deform a CAD solid to match *targets* while preserving topology continuity.

    Parameters
    ----------
    solid:
        The root :class:`~cadmap.geometry.topology.Topology` node (faces as
        ``solid.descendent``).  Geometry (BsplineSurface, Curve2d) must already
        be attached via :func:`~cadmap.conversion.occ.add_geom_to_solid`.
    targets:
        List of (M_i, 3) arrays — one point-cloud patch per surface.
    device:
        ``"cpu"`` or ``"cuda"``.
    lr:
        Initial learning rate for SGD.
    n_epochs:
        Number of optimisation epochs.
    sample_size:
        Number of edge samples for the constraint matrix.
    lr_milestones_step:
        Milestone spacing for the multi-step LR scheduler.
    lr_gamma:
        LR decay factor at each milestone.
    """

    def __init__(
        self,
        solid: Topology,
        targets: List[np.ndarray],
        device: str = "cpu",
        lr: float = 10.0,
        n_epochs: int = 1000,
        sample_size: int = 15,
        lr_milestones_step: int = 1000,
        lr_gamma: float = 0.5,
    ) -> None:
        self.solid = solid
        self.targets = targets
        self.device = device
        self.n_epochs = n_epochs
        self.sample_size = sample_size

        # --- Flatten all control points ---
        ctpall = np.concatenate([f.geom.init_ctp.reshape(-1, 3) for f in solid.descendent])
        weightall = np.concatenate([f.geom.weight.reshape(-1) for f in solid.descendent])

        self.inpCtrlPts_all = torch.nn.Parameter(
            torch.from_numpy(copy.deepcopy(ctpall)).to(dtype=torch.float32, device=device)
        )
        self.inpWeight_all = torch.nn.Parameter(
            torch.from_numpy(copy.deepcopy(weightall)).to(dtype=torch.float32, device=device)
        )

        self.opt = torch.optim.SGD([self.inpCtrlPts_all], lr=lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.opt,
            milestones=list(np.arange(lr_milestones_step, n_epochs * 40, lr_milestones_step)),
            gamma=lr_gamma,
        )

        # --- Constraint matrix ---
        self.all_basis, self.edge_map, self.periodic_map = build_constraint_matrix(
            solid, ctpall, sample_size=sample_size
        )
        self.unique_vertices = get_unique_vertices(solid)

        # --- History ---
        self.loss_history: List[float] = []
        self.saved_ctrl_pts: List[torch.Tensor] = [self.inpCtrlPts_all.detach().cpu().clone()]

    # ------------------------------------------------------------------

    def _compute_surface_loss(self, face, active_faces, local_norm_basis_tensor) -> torch.Tensor:
        surf3d = face.geom
        surf_i = int(surf3d.surf_id.split("_")[-1])
        weight = torch.tensor(surf3d.weight, device=self.device, dtype=torch.float32).unsqueeze(-1)
        inpCtrlPts = self.inpCtrlPts_all[surf3d.ctp_index[0]: surf3d.ctp_index[1]]
        inpCtrlPts = inpCtrlPts.view(surf3d.size_u, surf3d.size_v, 3)

        out = surf3d.DiffSurf(
            torch.cat(((inpCtrlPts * weight).unsqueeze(0), weight.unsqueeze(0)), dim=-1)
        ).squeeze(0)
        out_inside = out[surf3d.inside_evalpts, :]

        target = torch.tensor(self.targets[surf_i], dtype=torch.float32)
        Elements = torch.tensor(
            create_customized_trimesh(surf3d.inside_evalpts), dtype=torch.int
        )

        _, _, _, min_distance = find_closest_coord_on_mesh(
            out_inside, Elements, target, device=self.device
        )
        n_pts = target.shape[0]
        return sum(min_distance) / n_pts

    def _project_constraints(self, local_norm_basis_tensor: torch.Tensor, trainable_indices: List[int]) -> None:
        with torch.no_grad():
            mask = torch.zeros_like(self.inpCtrlPts_all.grad, dtype=torch.bool)
            mask[trainable_indices] = True
            self.inpCtrlPts_all.grad.data[~mask] = 0.0

        if local_norm_basis_tensor.shape[0] > 0:
            with torch.no_grad():
                for xi in range(3):
                    vec = self.inpCtrlPts_all.data[..., xi]
                    projections = torch.mv(local_norm_basis_tensor, vec)
                    correction = torch.mv(local_norm_basis_tensor.T, projections)
                    self.inpCtrlPts_all.data[..., xi] -= correction

    def run(self) -> List[torch.Tensor]:
        """Run the optimisation loop and return the list of saved control-point snapshots."""
        pbar = tqdm(range(self.n_epochs))

        for epoch in pbar:
            total_epoch_loss = 0.0

            for vertex in self.unique_vertices:
                active_faces, active_edges = get_connectivity_of_vertex(vertex, self.solid)
                fixed_edges = get_edges_of_faces(active_faces, exclude_edges=active_edges)
                trainable_indices = get_cp_indices_of_faces(active_faces)

                all_local_edge_rows = get_constraint_rows_for_edges(
                    active_edges + fixed_edges,
                    self.edge_map,
                    self.periodic_map,
                    active_faces,
                )
                local_norm_basis = compute_local_basis(self.all_basis, all_local_edge_rows)
                local_norm_basis_tensor = torch.tensor(
                    local_norm_basis, device=self.device, dtype=torch.float32
                )

                def closure():
                    self.opt.zero_grad()
                    loss = sum(
                        self._compute_surface_loss(face, active_faces, local_norm_basis_tensor)
                        for face in active_faces
                    )
                    loss.backward(retain_graph=True)
                    return loss

                step_loss = self.opt.step(closure)
                loss_val = step_loss.item() if hasattr(step_loss, "item") else 0.0
                total_epoch_loss += loss_val

                active_rows = get_constraint_rows_for_edges(
                    active_edges, self.edge_map, self.periodic_map, active_faces
                )
                active_basis = compute_local_basis(self.all_basis, active_rows)
                active_tensor = torch.tensor(active_basis, device=self.device, dtype=torch.float32)
                self._project_constraints(active_tensor, trainable_indices)

            avg_loss = total_epoch_loss / max(len(self.unique_vertices), 1)
            self.loss_history.append(avg_loss)
            self.scheduler.step()
            self.saved_ctrl_pts.append(self.inpCtrlPts_all.detach().cpu().clone())
            pbar.set_description(f"Epoch {epoch + 1}, Loss: {avg_loss:.6f}")

        return self.saved_ctrl_pts

    def export_occ_solid(self):
        """Convert the current control points back to an OCC solid."""
        from cadmap.visualization.plots import convert_to_OCC_solid
        occ_solid, _, _ = convert_to_OCC_solid(self.solid, self.inpCtrlPts_all)
        return occ_solid
