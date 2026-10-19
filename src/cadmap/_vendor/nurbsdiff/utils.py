import numpy as np
import torch


def gen_knot_vector(p,n, delta=1e-6):
    # p: degree, n: number of control points; m+1: number of knots
    m = p + n + 1

    # Calculate a uniform interval for middle knots
    num_segments = (m - 2*(p+1) + 1)  # number of segments in the middle

    # First degree+1 knots are "knot_min"
    knot_vector = [(j+1)*delta for j in range(0, p)]

    # Middle knots
    knot_vector += [mid_knot for mid_knot in np.linspace(delta*(p+1), 1, num_segments+1)]

    # Last degree+1 knots are "knot_max"
    knot_vector += [float(1) for j in range(0, p)]

    # Return auto-generated knot vector
    return knot_vector


import torch

def find_span_torch(n, p, u, U, eps=1e-5):
    """
    Find the knot span index for parameter u.
    Algorithm A2.1 from The NURBS Book (page 68), adapted for PyTorch.

    Args:
        n: number of control points - 1 (for 7 CPs, n=6)
        p: degree (2)
        u: parameter value(s)
        U: knot vector tensor
        eps: tolerance for boundary checks only
    """
    # 1. Handle batch of u values
    if isinstance(u, torch.Tensor) and u.dim() > 0:
        spans = torch.zeros_like(u, dtype=torch.long)
        for i in range(u.shape[0]):
            spans[i] = find_span_torch(n, p, u[i].item(), U, eps)
        return spans

    u_val = u.item() if isinstance(u, torch.Tensor) else u

    # 2. Upper Bound Check (CRITICAL FIX)
    # If u is at the end (1.0), we must return 'n' (the last valid span).
    # The previous code returned 'n-1', which for your knot vector is an empty span.
    if u_val >= U[n + 1] - eps:
        return n

    # 3. Lower Bound Check
    if u_val < U[p]:
        return p

    # 4. Binary Search
    low = p
    high = n + 1
    mid = (low + high) // 2

    # Standard Binary Search
    # We look for index 'mid' such that U[mid] <= u < U[mid+1]
    # REMOVED EPS here: The comparison must be strict to handle repeated knots correctly.
    while u_val < U[mid] or u_val >= U[mid + 1]:
        if u_val < U[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2

    return mid


def basis_funs_torch(i, u, p, U):
    """
    Compute the nonvanishing basis functions
    Algorithm A2.2 from The NURBS Book (page 70)

    Args:
        i: knot span index
        u: parameter value
        p: degree
        U: knot vector tensor

    Returns:
        N: tensor of shape (p+1,) containing basis function values
    """
    u_val = u.item() if isinstance(u, torch.Tensor) else u

    N = torch.zeros(p+1, device=U.device, dtype=U.dtype)
    left = torch.zeros(p+1, device=U.device, dtype=U.dtype)
    right = torch.zeros(p+1, device=U.device, dtype=U.dtype)

    N[0] = 1.0

    for j in range(1, p+1):
        left[j] = u_val - U[i+1-j]
        right[j] = U[i+j] - u_val
        saved = 0.0

        for r in range(j):
            temp = N[r] / (right[r+1] + left[j-r])
            N[r] = saved + right[r+1] * temp
            saved = left[j-r] * temp

        N[j] = saved

    return N