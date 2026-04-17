"""Tests for cadmap.deformation.mesh_utils."""

import numpy as np
import pytest
import torch

from cadmap.deformation.mesh_utils import (
    create_customized_trimesh,
    Barycentric_coordinates_on_triangle,
    find_closest_coord_on_mesh,
)


class TestCreateTrimesh:
    def test_returns_triangle_array(self):
        mask = np.ones((4, 4), dtype=bool)
        elements = create_customized_trimesh(mask)
        assert isinstance(elements, np.ndarray)
        assert elements.ndim == 2
        assert elements.shape[1] == 3

    def test_partial_mask(self):
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 1] = True
        mask[1, 2] = True
        mask[2, 1] = True
        elements = create_customized_trimesh(mask)
        assert elements.ndim == 2


class TestBarycentricCoords:
    def test_centroid_has_equal_weights(self):
        # Batched interface: (B, 3, 3) triangles and (B, 3) points
        A = torch.tensor([0.0, 0.0, 0.0])
        B = torch.tensor([1.0, 0.0, 0.0])
        C = torch.tensor([0.0, 1.0, 0.0])
        centroid = ((A + B + C) / 3).unsqueeze(0)  # (1, 3)
        tri = torch.stack([A, B, C]).unsqueeze(0)   # (1, 3, 3)
        lam = Barycentric_coordinates_on_triangle(tri, centroid)  # (1, 3)
        assert lam.shape == (1, 3)
        assert abs(float(lam[0, 0]) - 1 / 3) < 1e-5
        assert abs(float(lam[0, 1]) - 1 / 3) < 1e-5
        assert abs(float(lam[0, 2]) - 1 / 3) < 1e-5


class TestFindClosestOnMesh:
    def test_basic(self):
        # 2 triangles forming a unit square; use knn_k=3 (< 4 vertices)
        verts = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int)
        query = torch.tensor([[0.5, 0.5, 1.0]])
        _, _, _, dist = find_closest_coord_on_mesh(verts, faces, query, knn_k=3)
        assert dist[0].item() < 2.0
