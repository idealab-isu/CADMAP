"""Tests for cadmap.metrics — Chamfer and Hausdorff distances."""

import numpy as np
import pytest
import torch

from cadmap.metrics.distances import (
    chamfer_distance_one_side,
    Hausdorff_distance_cust,
    Hausdorff_distance_one_side,
)

DEVICE = "cpu"


def _cloud(n: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.random((n, 3)).astype(np.float32))


class TestChamfer:
    def test_identical_clouds_near_zero(self):
        x = _cloud(50)
        d = chamfer_distance_one_side(x, x, face=None, device=DEVICE)
        assert float(d) < 1e-4

    def test_shifted_cloud_positive(self):
        x = _cloud(50)
        y = x + 1.0
        d = chamfer_distance_one_side(x, y, face=None, device=DEVICE)
        assert float(d) > 0

    def test_returns_scalar(self):
        x = _cloud(30, seed=1)
        y = _cloud(40, seed=2)
        d = chamfer_distance_one_side(x, y, face=None, device=DEVICE)
        assert d.ndim == 0


class TestHausdorff:
    def test_identical_clouds_near_zero(self):
        x = _cloud(50)
        d = Hausdorff_distance_cust(x, x, device=DEVICE)
        assert float(d) < 1e-4

    def test_one_side_leq_symmetric(self):
        x = _cloud(30, seed=3)
        y = _cloud(30, seed=4)
        d_sym = float(Hausdorff_distance_cust(x, y, device=DEVICE))
        d_one = float(Hausdorff_distance_one_side(x, y, device=DEVICE))
        assert d_sym >= 0 and d_one >= 0
