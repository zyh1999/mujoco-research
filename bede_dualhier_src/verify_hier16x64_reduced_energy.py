#!/usr/bin/env python3
import torch


def main():
    torch.manual_seed(7)
    dtype = torch.float64
    samples, group_size, params = 1024, 64, 37
    num_groups = samples // group_size
    damping = 0.03

    h = torch.randn(samples, params, dtype=dtype)
    ratio = torch.exp(0.2 * torch.randn(samples, dtype=dtype))
    target = torch.randn(samples, dtype=dtype)
    grouped_h = h.view(num_groups, group_size, params)
    grouped_ratio = ratio.view(num_groups, group_size)
    grouped_target = target.view(num_groups, group_size)

    local_kernel = torch.bmm(grouped_h, grouped_h.transpose(1, 2)) / samples
    eye = torch.eye(group_size, dtype=dtype).expand(num_groups, -1, -1)
    local_system = local_kernel * grouped_ratio.unsqueeze(1) + damping * eye
    local_alpha = torch.linalg.solve(local_system, grouped_target.unsqueeze(-1)).squeeze(-1)

    weighted_alpha = grouped_ratio * local_alpha
    basis = torch.bmm(grouped_h.transpose(1, 2), weighted_alpha.unsqueeze(-1)).squeeze(-1)
    metric = (grouped_ratio * local_alpha.square()).sum(dim=1)
    rhs = (grouped_ratio * local_alpha * grouped_target).sum(dim=1)
    reduced = basis @ basis.t() / samples + damping * torch.diag(metric)

    coefficient_basis = torch.zeros(samples, num_groups, dtype=dtype)
    for group in range(num_groups):
        start = group * group_size
        coefficient_basis[start : start + group_size, group] = local_alpha[group]

    d_ratio = torch.diag(ratio)
    kernel = h @ h.t() / samples
    full_energy = d_ratio @ kernel @ d_ratio + damping * d_ratio
    explicit_reduced = coefficient_basis.t() @ full_energy @ coefficient_basis
    explicit_rhs = coefficient_basis.t() @ d_ratio @ target

    rho = torch.linalg.solve(reduced, rhs)
    update = basis.t() @ rho / samples
    explicit_update = h.t() @ d_ratio @ coefficient_basis @ rho / samples

    errors = {
        "reduced_system_max_abs": (reduced - explicit_reduced).abs().max().item(),
        "reduced_rhs_max_abs": (rhs - explicit_rhs).abs().max().item(),
        "parameter_map_max_abs": (update - explicit_update).abs().max().item(),
    }
    for name, value in errors.items():
        print(f"{name}={value:.3e}")
        if value > 1e-9:
            raise AssertionError(f"{name} too large: {value}")
    print("HIER16X64_REDUCED_ENERGY_OK")


if __name__ == "__main__":
    main()
