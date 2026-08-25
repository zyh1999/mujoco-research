"""Detached residual diagnostics for the frozen exact score-kernel systems."""

from __future__ import annotations

import torch


@torch.no_grad()
def detached_exact_score_solver_residual(
    h: torch.Tensor,
    rhs: torch.Tensor,
    solution: torch.Tensor,
    damping: float,
    *,
    ratio: torch.Tensor | None = None,
    previous_projection: torch.Tensor | None = None,
    epsilon: float = 1.0e-12,
) -> tuple[float, float, float]:
    """Return ||A x - b||_2 / max(||b||_2, epsilon), numerator, denominator.

    This reconstructs the already-solved frozen system after ``solution`` has
    been fixed.  It is detached, contains no random operations, and never
    mutates the supplied tensors or any optimizer/training state.

    For m score/Jacobian rows, K = H H^T / m.  The frozen actor system uses
    A = K D_ratio + damping I and b = rhs - previous_projection.  The frozen
    critic system omits ratio and previous_projection, so A = K + damping I
    and b = rhs.
    """
    if h.ndim != 2:
        raise ValueError(f"h must be 2D, got {tuple(h.shape)}")
    if rhs.ndim != 1 or solution.ndim != 1:
        raise ValueError("rhs and solution must be one-dimensional")
    if h.shape[0] != rhs.numel() or rhs.shape != solution.shape:
        raise ValueError("incompatible score-system shapes")

    matrix = h.detach() @ h.detach().t() / float(h.shape[0])
    effective_rhs = rhs.detach().to(dtype=matrix.dtype)
    if previous_projection is not None:
        effective_rhs = effective_rhs - previous_projection.detach().to(dtype=matrix.dtype)
    if ratio is not None:
        matrix = matrix * ratio.detach().to(dtype=matrix.dtype).unsqueeze(0)
    matrix = matrix + float(damping) * torch.eye(
        matrix.shape[0], device=matrix.device, dtype=matrix.dtype
    )

    residual = matrix.mv(solution.detach().to(dtype=matrix.dtype)) - effective_rhs
    numerator = residual.norm()
    denominator = effective_rhs.norm().clamp_min(float(epsilon))
    relative = numerator / denominator
    return float(relative.item()), float(numerator.item()), float(denominator.item())
