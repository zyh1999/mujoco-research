#!/usr/bin/env python3
"""Fixed-input proof that residual telemetry does not alter training state."""

from __future__ import annotations

import copy
import json
import math

import torch

from ktrue_momentum_stage.solver_residual_telemetry import (
    detached_exact_score_solver_residual,
)


def tensors_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(tensors_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            tensors_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def main() -> int:
    torch.manual_seed(20260825)
    actor = torch.nn.Linear(5, 3)
    critic = torch.nn.Linear(5, 1)
    actor_optimizer = torch.optim.SGD(actor.parameters(), lr=0.05, momentum=0.7)
    critic_optimizer = torch.optim.SGD(critic.parameters(), lr=0.1, momentum=0.7)

    obs = torch.randn(16, 5)
    actor_target = torch.randn(16, 3)
    critic_target = torch.randn(16, 1)
    actor_loss = torch.nn.functional.mse_loss(actor(obs), actor_target)
    critic_loss = torch.nn.functional.mse_loss(critic(obs), critic_target)
    actor_loss.backward()
    critic_loss.backward()
    actor_optimizer.step()
    critic_optimizer.step()

    h_actor = torch.randn(8, 9)
    h_critic = torch.randn(8, 7)
    rhs_actor = torch.randn(8)
    rhs_critic = torch.randn(8)
    ratio = torch.rand(8).add_(0.5)
    previous_projection = torch.randn(8)
    damping = 0.03
    eye = torch.eye(8)
    actor_matrix = (h_actor @ h_actor.t() / 8.0) * ratio.unsqueeze(0) + damping * eye
    critic_matrix = h_critic @ h_critic.t() / 8.0 + damping * eye
    actor_solution = torch.linalg.solve(actor_matrix, rhs_actor - previous_projection)
    critic_solution = torch.linalg.solve(critic_matrix, rhs_critic)
    direction_actor = h_actor.t().mv(ratio * actor_solution) / 8.0
    direction_critic = h_critic.t().mv(critic_solution) / 8.0
    loss_value = float(actor_loss.detach())
    kl_value = float((actor(obs).detach() - actor_target).square().mean())
    vf_value = float(critic_loss.detach())

    before = {
        "actor_parameters": copy.deepcopy(actor.state_dict()),
        "critic_parameters": copy.deepcopy(critic.state_dict()),
        "actor_optimizer": copy.deepcopy(actor_optimizer.state_dict()),
        "critic_optimizer": copy.deepcopy(critic_optimizer.state_dict()),
        "actor_direction": direction_actor.clone(),
        "critic_direction": direction_critic.clone(),
        "previous_projection": previous_projection.clone(),
        "rng_state": torch.random.get_rng_state().clone(),
        "loss": loss_value,
        "kl": kl_value,
        "vf": vf_value,
    }

    actor_residual = detached_exact_score_solver_residual(
        h_actor,
        rhs_actor,
        actor_solution,
        damping,
        ratio=ratio,
        previous_projection=previous_projection,
    )
    critic_residual = detached_exact_score_solver_residual(
        h_critic, rhs_critic, critic_solution, damping
    )

    after = {
        "actor_parameters": actor.state_dict(),
        "critic_parameters": critic.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "actor_direction": direction_actor,
        "critic_direction": direction_critic,
        "previous_projection": previous_projection,
        "rng_state": torch.random.get_rng_state(),
        "loss": loss_value,
        "kl": kl_value,
        "vf": vf_value,
    }
    checks = {key: tensors_equal(before[key], after[key]) for key in before}
    checks["actor_residual_finite"] = all(math.isfinite(value) for value in actor_residual)
    checks["critic_residual_finite"] = all(math.isfinite(value) for value in critic_residual)
    checks["all_bitwise_equal"] = all(checks[key] for key in before)
    checks["pass"] = all(checks.values())
    print(json.dumps({
        "checks": checks,
        "actor_residual": actor_residual,
        "critic_residual": critic_residual,
    }, sort_keys=True))
    return 0 if checks["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
