#!/usr/bin/env python3
import torch


def main():
    torch.manual_seed(19)
    dtype = torch.float64
    samples, group_size, params = 1024, 64, 43
    num_groups = samples // group_size
    damping = 0.03
    vf_coef = 1.7

    h = torch.randn(samples, params, dtype=dtype)
    j = torch.randn(samples, params, dtype=dtype)
    ratio = torch.exp(0.2 * torch.randn(samples, dtype=dtype))
    actor_target = torch.randn(samples, dtype=dtype)
    critic_residual = torch.randn(samples, dtype=dtype)
    critic_features = vf_coef**0.5 * j
    critic_target = vf_coef**0.5 * critic_residual

    grouped_h = h.view(num_groups, group_size, params)
    grouped_ratio = ratio.view(num_groups, group_size)
    grouped_actor_target = actor_target.view(num_groups, group_size)
    eye = torch.eye(group_size, dtype=dtype).expand(num_groups, -1, -1)

    actor_local_kernel = torch.bmm(grouped_h, grouped_h.transpose(1, 2)) / samples
    actor_local_system = actor_local_kernel * grouped_ratio.unsqueeze(1) + damping * eye
    actor_alpha = torch.linalg.solve(
        actor_local_system, grouped_actor_target.unsqueeze(-1)
    ).squeeze(-1)
    actor_basis = torch.bmm(
        grouped_h.transpose(1, 2), (grouped_ratio * actor_alpha).unsqueeze(-1)
    ).squeeze(-1)
    actor_metric = (grouped_ratio * actor_alpha.square()).sum(dim=1)
    actor_rhs = (grouped_ratio * actor_alpha * grouped_actor_target).sum(dim=1)

    grouped_c = critic_features.view(num_groups, group_size, params)
    grouped_critic_target = critic_target.view(num_groups, group_size)
    critic_local_kernel = torch.bmm(grouped_c, grouped_c.transpose(1, 2)) / samples
    critic_local_system = critic_local_kernel + damping * eye
    critic_alpha = torch.linalg.solve(
        critic_local_system, grouped_critic_target.unsqueeze(-1)
    ).squeeze(-1)
    critic_basis = torch.bmm(
        grouped_c.transpose(1, 2), critic_alpha.unsqueeze(-1)
    ).squeeze(-1)
    critic_metric = critic_alpha.square().sum(dim=1)
    critic_rhs = (critic_alpha * grouped_critic_target).sum(dim=1)

    basis = torch.cat([actor_basis, critic_basis], dim=0)
    metric = torch.cat([actor_metric, critic_metric], dim=0)
    rhs = torch.cat([actor_rhs, critic_rhs], dim=0)
    reduced = basis @ basis.t() / samples + damping * torch.diag(metric)

    coefficient_basis = torch.zeros(2 * samples, 2 * num_groups, dtype=dtype)
    for group in range(num_groups):
        start = group * group_size
        coefficient_basis[start : start + group_size, group] = actor_alpha[group]
        critic_start = samples + start
        coefficient_basis[
            critic_start : critic_start + group_size, num_groups + group
        ] = critic_alpha[group]

    actor_features = ratio.unsqueeze(1) * h
    full_features = torch.cat([actor_features, critic_features], dim=0)
    full_metric = torch.cat([ratio, torch.ones(samples, dtype=dtype)], dim=0)
    full_target = torch.cat([ratio * actor_target, critic_target], dim=0)
    full_energy = full_features @ full_features.t() / samples
    full_energy = full_energy + damping * torch.diag(full_metric)
    explicit_reduced = coefficient_basis.t() @ full_energy @ coefficient_basis
    explicit_rhs = coefficient_basis.t() @ full_target

    rho = torch.linalg.solve(reduced, rhs)
    update = basis.t() @ rho / samples
    explicit_update = full_features.t() @ coefficient_basis @ rho / samples

    errors = {
        "joint_reduced_system_max_abs": (reduced - explicit_reduced).abs().max().item(),
        "joint_reduced_rhs_max_abs": (rhs - explicit_rhs).abs().max().item(),
        "joint_parameter_map_max_abs": (update - explicit_update).abs().max().item(),
    }
    for name, value in errors.items():
        print(f"{name}={value:.3e}")
        if value > 1e-9:
            raise AssertionError(f"{name} too large: {value}")
    print("DUALHIER16X64_TOP32_REDUCED_ENERGY_OK")


if __name__ == "__main__":
    main()
