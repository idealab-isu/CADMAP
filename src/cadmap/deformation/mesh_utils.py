"""Mesh utilities: Delaunay trimesh, barycentric coordinates, closest-point search."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from cadmap._internal.math_ops import KNearestNeighbors


def create_customized_trimesh(in_trim_array: np.ndarray) -> np.ndarray:
    """Build a Delaunay triangulation over the interior grid points of a surface.

    Parameters
    ----------
    in_trim_array:
        Boolean array of shape ``(nu, nv)`` marking which UV grid points are
        inside the trim region.

    Returns
    -------
    Elements of shape ``(T, 3)``: triangle vertex indices into the list of
    *in-trim* grid points.
    """
    from scipy.spatial import Delaunay

    x, y = np.where(in_trim_array)
    pts = np.stack([x, y]).T
    tri = Delaunay(pts)
    return tri.simplices


def create_mesh_o3d(vertices: np.ndarray, faces: np.ndarray):
    """Wrap vertices and faces into an Open3D ``TriangleMesh``."""
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    return mesh


def Barycentric_coordinates_on_triangle(
    triangle_vertices: torch.Tensor,
    point: torch.Tensor,
) -> torch.Tensor:
    """Barycentric coordinates of *point* w.r.t. *triangle_vertices*.

    Parameters
    ----------
    triangle_vertices:
        ``(B, 3, 3)`` — B triangles, 3 vertices each, 3 XYZ coords.
    point:
        ``(B, 3)`` — B query points.

    Returns
    -------
    ``(B, 3)`` barycentric weights λ₀, λ₁, λ₂.
    """
    edges = torch.roll(triangle_vertices, shifts=-1, dims=1) - triangle_vertices
    vecs = triangle_vertices - point.unsqueeze(1)
    area_total = torch.linalg.cross(edges[:, 0], edges[:, 1]).norm(dim=1) * 0.5
    area_sub = torch.linalg.cross(
        torch.roll(vecs, shifts=-1, dims=1),
        torch.roll(vecs, shifts=1, dims=1),
        dim=2,
    ).norm(dim=2) * 0.5
    return area_sub / area_total.unsqueeze(1)


def find_closest_coord_on_mesh(
    mesh: torch.Tensor,
    Elements: torch.Tensor,
    targets: torch.Tensor,
    device: str = "cpu",
    knn_k: int = 15,
) -> Tuple[List, List, List, List]:
    """For each target point find the closest point on the mesh.

    Parameters
    ----------
    mesh:
        ``(V, 3)`` tensor of mesh vertex positions.
    Elements:
        ``(T, 3)`` integer tensor of triangle vertex indices.
    targets:
        ``(N, 3)`` tensor of query points.
    device:
        Compute device.
    knn_k:
        Number of nearest mesh vertices used as candidate triangles.

    Returns
    -------
    (closest_BayLambdas, closest_vertices_index, closest_point, min_distance)
    Each is a list of length N.
    """
    closest_BayLambdas: List = []
    closest_vertices_index: List = []
    closest_point_list: List = []
    min_distance: List = []

    targets_dev = targets.to(device)

    # triangle edges and normals
    tri_verts = mesh[Elements]                              # (T, 3, 3)
    tri_edges = torch.roll(tri_verts, -1, dims=1) - tri_verts  # (T, 3, 3)
    mesh_normals = torch.linalg.cross(tri_edges[:, 0], tri_edges[:, 1])  # (T, 3)

    knn = KNearestNeighbors(n_neighbors=knn_k)
    knn.fit(mesh)
    _, nn_indices = knn.kneighbors(targets_dev)             # (N, knn_k)
    nn_indices = nn_indices.cpu()

    normals_dot = torch.matmul(mesh_normals, mesh_normals.T).cpu()  # (T, T)
    cos_thresh = float(np.cos(np.pi / 3))

    for pt_i in range(targets_dev.shape[0]):
        # collect candidate triangles touching the knn_k closest vertices
        row_elem_idx = torch.cat([
            torch.where(Elements == x)[0] for x in nn_indices[pt_i]
        ])
        first_n = nn_indices[pt_i, 0:1]
        first_elem = torch.where(Elements == first_n[0])[0]
        first_count = first_elem.shape[0]

        # filter out back-facing / geometrically redundant triangles
        group = np.arange(first_count)
        group_elem = row_elem_idx[group]
        for ti in range(first_count, row_elem_idx.shape[0]):
            if torch.all(normals_dot[row_elem_idx[ti], group_elem] < cos_thresh):
                group = np.append(group, ti)
                group_elem = row_elem_idx[group]
        cand_elem = row_elem_idx[group]

        verts = mesh[Elements[cand_elem]]                   # (C, 3, 3)
        pt = targets_dev[pt_i]

        normals_cand = mesh_normals[cand_elem]              # (C, 3)
        p2plane = pt - verts[:, 0]
        dists_proj = torch.matmul(p2plane, normals_cand.T).diagonal()
        proj_pts = pt - normals_cand * dists_proj.unsqueeze(1)

        lambdas = Barycentric_coordinates_on_triangle(verts, proj_pts)
        in_tri = torch.all(
            torch.stack([
                torch.all(lambdas >= 0, 1),
                torch.isclose(lambdas.sum(1), torch.ones(lambdas.shape[0])),
            ]), 0
        )

        # edge projections for out-of-triangle fallback
        vecs = pt - verts
        en = torch.linalg.norm(tri_edges[cand_elem], dim=-1, keepdim=True)
        t_proj = torch.clamp(torch.sum(vecs * tri_edges[cand_elem], dim=-1, keepdim=True) / en, 0, 1)
        edge_proj = verts + t_proj * tri_edges[cand_elem]
        edge_dists = torch.linalg.norm(pt - edge_proj, dim=-1)
        out_min, out_argmin = edge_dists.min(dim=-1)
        out_pts = edge_proj[torch.arange(edge_proj.shape[0]), out_argmin]
        out_lam = Barycentric_coordinates_on_triangle(verts, out_pts)

        all_lam = torch.cat([lambdas[in_tri], out_lam])
        all_idx = torch.cat([cand_elem[in_tri], cand_elem])
        all_pts = torch.cat([proj_pts[in_tri], out_pts])
        all_d = torch.cat([dists_proj[in_tri].abs(), out_min])

        best = torch.argmin(all_d)
        closest_BayLambdas.append(all_lam[best])
        closest_vertices_index.append(all_idx[best])
        closest_point_list.append(all_pts[best])
        min_distance.append(all_d[best])

    return closest_BayLambdas, closest_vertices_index, closest_point_list, min_distance
