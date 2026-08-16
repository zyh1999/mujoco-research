"""Same-policy actor-update direction diagnostics for MuJoCo experiments.

This module intentionally compares raw ascent directions before learning-rate,
momentum, or scalar clipping.  It is used from a one-shot probe only.
"""
import json
from pathlib import Path

import torch

from utils.cg import conjugate_gradient
from utils.sketching import solve_score_kernel_system


METHODS = (
    "ppo",
    "emp256",
    "emp256_fullrhs",
    "energyfree255p1",
    "full_empirical_direct",
    "true_fisher_cg",
    "kfac_hybrid",
    "kfac_corrected",
)


def _cosine(directions):
    stacked = torch.stack([directions[name] for name in METHODS])
    stacked = stacked / stacked.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return stacked @ stacked.t()


def save_probe(path, payload, directions):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["methods"] = list(METHODS)
    payload["norms"] = {name: float(direction.norm().item()) for name, direction in directions.items()}
    payload["cosine"] = _cosine(directions).detach().cpu().tolist()
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    torch.save({"methods": METHODS, "directions": {k: v.detach().cpu() for k, v in directions.items()}}, path.with_suffix(".pt"))


def empirical_direction(scores, targets, ratios, damping):
    """Exact score-kernel map H^T D alpha / m, matching the RAT solver."""
    alpha = solve_score_kernel_system(scores, targets, damping, ratio=ratios,
                                      kernel="exact", normalization="none")
    return scores.t().mv(ratios * alpha) / float(scores.shape[0])


def empirical_subsample_fullrhs_direction(scores, full_rhs, ratios, damping):
    """Use subsampled empirical Fisher curvature with the full-batch RHS."""
    count = float(scores.shape[0])
    kernel = scores @ scores.t()
    weighted_kernel = kernel * ratios.to(dtype=kernel.dtype).unsqueeze(0)
    eye = torch.eye(scores.shape[0], device=scores.device, dtype=kernel.dtype)
    beta = torch.linalg.solve(
        weighted_kernel + count * damping * eye,
        scores.mv(full_rhs.to(dtype=scores.dtype)),
    )
    correction = scores.t().mv(ratios.to(dtype=scores.dtype) * beta)
    return (full_rhs - correction.to(dtype=full_rhs.dtype)) / damping


def full_empirical_cg(scores, rhs, ratios, damping, steps):
    count = float(scores.shape[0])
    def fvp(vector):
        return scores.t().mv(ratios * scores.mv(vector)) / count + damping * vector
    direction = conjugate_gradient(fvp, rhs, nsteps=steps)
    residual = (fvp(direction) - rhs).norm() / rhs.norm().clamp_min(1e-12)
    return direction, float(residual.item())


def energyfree_255p1_direction(full_scores, targets, ratios, anchor_indices, damping):
    """The existing anchor-mode 255+1 energy solve, expressed as a raw step."""
    output_dtype = full_scores.dtype
    # The Schur denominator is a difference of two nearly equal quadratic
    # forms. Evaluate this small solve in float64 so its sign is meaningful.
    full_scores = full_scores.to(dtype=torch.float64)
    targets = targets.to(dtype=torch.float64)
    ratios = ratios.to(dtype=torch.float64)
    total = int(full_scores.shape[0])
    anchors = anchor_indices
    mask = torch.ones(total, dtype=torch.bool, device=full_scores.device)
    mask[anchors] = False
    nonanchors = torch.arange(total, device=full_scores.device)[mask]
    hs, bs, rs = full_scores[anchors], targets[anchors], ratios[anchors]
    hn, bn, rn = full_scores[nonanchors], targets[nonanchors], ratios[nonanchors]
    size = float(hs.shape[0])
    kernel = hs @ hs.t() / size
    system = kernel * rs.unsqueeze(0) + damping * torch.eye(hs.shape[0], device=hs.device, dtype=hs.dtype)
    base = hn.t().mv(rn * bn)
    coupling = hs.mv(base) / size
    u = torch.linalg.solve(system, bs)
    v = torch.linalg.solve(system, coupling)
    d0 = hs.t().mv(rs * u) / float(total)
    d1 = (base - hs.t().mv(rs * v)) / float(total)
    target_energy = bn.dot(bn)
    # This is the bottom-right block of the reduced 256-DOF system, whose
    # kernel normalization is the 255-row anchor-system size.  The final
    # parameter-space map below still uses the full minibatch denominator.
    c_base = base.dot(base) / size
    schur_num = target_energy - coupling.dot(u)
    schur_den = c_base + damping * target_energy - coupling.dot(v)
    # Preserve the Schur-complement sign. clamp_min() would turn every
    # negative denominator into +eps and create an enormous spurious rho.
    schur_den_safe = torch.where(
        schur_den >= 0,
        schur_den.abs().clamp_min(1e-12),
        -schur_den.abs().clamp_min(1e-12),
    )
    rho = schur_num / schur_den_safe
    direction = (d0 + rho * d1).to(dtype=output_dtype)
    diagnostics = {
        "schur_num": float(schur_num.item()),
        "schur_den": float(schur_den.item()),
        "target_energy": float(target_energy.item()),
        "c_base": float(c_base.item()),
        "coupling_u": float(coupling.dot(u).item()),
        "coupling_v": float(coupling.dot(v).item()),
        "solve_dtype": "float64",
    }
    return direction, float(rho.item()), diagnostics


def true_fisher_cg(fvp, rhs, steps):
    """CG wrapper for a detached-reference exact-KL Hessian-vector product."""
    direction = conjugate_gradient(fvp, rhs, nsteps=steps)
    residual = (fvp(direction) - rhs).norm() / rhs.norm().clamp_min(1e-12)
    return direction, float(residual.item())


def kfac_snapshot_direction(model, optimizer_factory, fisher_loss, objective_loss):
    """Return a real K-FAC ascent displacement and restore model parameters."""
    saved = [parameter.detach().clone() for parameter in model.parameters() if parameter.requires_grad]
    before = torch.cat([parameter.flatten() for parameter in saved])
    optimizer = optimizer_factory(model)
    try:
        optimizer.zero_grad()
        optimizer.acc_stats = True
        fisher_loss.backward(retain_graph=True)
        optimizer.acc_stats = False
        optimizer.zero_grad()
        objective_loss.backward()
        optimizer.step()
        after = torch.cat([parameter.detach().flatten() for parameter in model.parameters() if parameter.requires_grad])
        return after - before
    finally:
        for parameter, value in zip((p for p in model.parameters() if p.requires_grad), saved):
            parameter.data.copy_(value)
