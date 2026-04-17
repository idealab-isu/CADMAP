"""Distance metrics for CAD deformation evaluation."""

from __future__ import annotations

import numpy as np
import torch
from torch.autograd import Variable


def chamfer_distance_one_side(
    pred,
    gt,
    face,
    device: str,
    side: int = 1,
) -> torch.Tensor:
    """One-sided Chamfer distance.

    Parameters
    ----------
    pred:
        Prediction array/tensor (N, 3).
    gt:
        Ground-truth array/tensor (M, 3).
    face:
        cadmap face Topology node (unused; kept for API compatibility).
    device:
        ``"cpu"`` or ``"cuda"``.
    side:
        0 → mean min distance from each *pred* to *gt*;
        1 → mean min distance from each *gt* to *pred*.
    """
    if isinstance(pred, np.ndarray):
        pred = Variable(torch.from_numpy(pred.astype(np.float32))).to(device)
    if isinstance(gt, np.ndarray):
        gt = Variable(torch.from_numpy(gt.astype(np.float32))).to(device)

    pred = pred.to(device)
    gt = gt.to(device)

    # diff[i, j] = squared distance between pred[i] and gt[j]
    diff = ((pred.unsqueeze(1) - gt.unsqueeze(0)) ** 2).sum(2)  # (N, M)

    if side == 0:
        return diff.min(1).values.mean()
    else:
        return diff.min(0).values.mean()


def Hausdorff_distance_cust(
    pred,
    gt,
    device: str,
    side: int = 1,
    topk: int = 10,
) -> torch.Tensor:
    """Custom Hausdorff: mean of the *topk* largest point-wise minimum distances."""
    if isinstance(pred, np.ndarray):
        pred = Variable(torch.from_numpy(pred.astype(np.float32))).to(device)
    if isinstance(gt, np.ndarray):
        gt = Variable(torch.from_numpy(gt.astype(np.float32))).to(device)

    pred = pred.to(device)
    gt = gt.to(device)

    diff = ((pred.unsqueeze(1) - gt.unsqueeze(0)) ** 2).sum(2)  # (N, M)

    k = min(topk, diff.shape[side])
    if side == 0:
        return torch.topk(diff.min(1).values, k)[0].mean()
    else:
        return torch.topk(diff.min(0).values, k)[0].mean()


def Hausdorff_distance_one_side(
    pred,
    gt,
    device: str,
    side: int = 1,
) -> torch.Tensor:
    """One-sided Hausdorff distance: max of min distances."""
    if isinstance(pred, np.ndarray):
        pred = Variable(torch.from_numpy(pred.astype(np.float32))).to(device)
    if isinstance(gt, np.ndarray):
        gt = Variable(torch.from_numpy(gt.astype(np.float32))).to(device)

    pred = pred.to(device)
    gt = gt.to(device)

    diff = ((pred.unsqueeze(1) - gt.unsqueeze(0)) ** 2).sum(2)  # (N, M)

    if side == 0:
        return diff.min(1).values.max()
    else:
        return diff.min(0).values.max()
