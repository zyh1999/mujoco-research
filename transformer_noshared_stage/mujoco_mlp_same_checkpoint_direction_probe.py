"""Compare MuJoCo actor directions at one PPO checkpoint and one fixed batch."""

import argparse
import copy
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap
from torch.optim import Adam

from stable_baselines3.common.env_util import make_vec_env

from kfac.kfac import KFACOptimizer
from utils.cg import conjugate_gradient
from utils.mujoco_direction_observer import (
    METHODS,
    empirical_direction,
    empirical_subsample_fullrhs_direction,
    energyfree_255p1_direction,
    save_probe,
    true_fisher_cg,
)
from utils.runners import Runner
from utils.utils import ActorCritic, build_mlp, set_seed
from vec_env import VecNormalize


ENV_IDS = {
    "ant": "Ant-v4",
    "halfcheetah": "HalfCheetah-v4",
    "hopper": "Hopper-v4",
    "humanoid": "Humanoid-v4",
    "humanoidstandup": "HumanoidStandup-v4",
    "swimmer": "Swimmer-v3",
    "walker2d": "Walker2d-v4",
}


class NetConfig:
    norm_obs = True
    a_dropout = 0.0
    c_dropout = 0.0
    a_hidden_size = 256
    c_hidden_size = 256
    a_num_layers = 2
    c_num_layers = 2


def flat_tensors(tensors):
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def selected_logp(outputs, actions):
    mu, logstd = outputs.chunk(2, dim=-1)
    return torch.distributions.Normal(mu, torch.exp(logstd)).log_prob(actions).sum(dim=-1)


def normalize_rollout(model, obs, ret, adv):
    if model.with_popart:
        model.last_v_layer.update(ret)
        ret = model.last_v_layer.normalize(ret)
        adv = model.last_v_layer.normalize(adv)
    if model.obs_rms is not None:
        model.obs_rms.training = True
        obs = model.obs_rms(obs)
        model.obs_rms.training = False
    return obs, ret, adv


def ppo_warmup(model, runner, updates, batch_size):
    pi_optim = Adam(model.pi_net.parameters(), lr=3e-4)
    v_optim = Adam(model.v_net.parameters(), lr=3e-4)
    total = runner.nsteps * runner.nenv
    indices = np.arange(total)

    for update in range(updates):
        model.eval()
        obs, ret, act, adv, _, epinfos = runner.run()
        obs, ret, adv = normalize_rollout(model, obs, ret, adv)
        with torch.no_grad():
            outputs_old = model.forward_pi(obs)
        model.train()

        for _ in range(4):
            np.random.shuffle(indices)
            for start in range(0, total, batch_size):
                mb = indices[start:start + batch_size]
                outputs = model.forward_pi(obs[mb])
                ratio = torch.exp(selected_logp(outputs, act[mb]) - selected_logp(outputs_old[mb], act[mb]))
                mb_adv = (adv[mb] - adv[mb].mean()) / (adv[mb].std() + 1e-8)
                unclipped = -ratio * mb_adv
                clipped = -torch.clamp(ratio, 0.8, 1.2) * mb_adv
                loss = torch.maximum(unclipped, clipped).mean()
                pi_optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.pi_net.parameters(), 0.5)
                pi_optim.step()

        for _ in range(4):
            np.random.shuffle(indices)
            for start in range(0, total, batch_size):
                mb = indices[start:start + batch_size]
                loss_v = F.mse_loss(model.forward_v(obs[mb]), ret[mb])
                v_optim.zero_grad()
                loss_v.backward()
                torch.nn.utils.clip_grad_norm_(model.v_net.parameters(), 5.0)
                v_optim.step()

        reward = np.mean([info["r"] for info in epinfos]) if epinfos else float("nan")
        print(f"PPO_WARMUP update={update + 1}/{updates} reward={reward:.6g}", flush=True)


def score_rows(model, obs, act):
    params = {name: value.detach() for name, value in model.pi_net.named_parameters()}
    buffers = {name: value.detach() for name, value in model.pi_net.named_buffers()}

    def one_logp(p, b, one_obs, one_act):
        outputs = functional_call(model.pi_net, (p, b), (one_obs.unsqueeze(0),))
        return selected_logp(outputs, one_act.unsqueeze(0)).squeeze(0)

    rows = vmap(grad(one_logp), in_dims=(None, None, 0, 0))(params, buffers, obs, act)
    return torch.cat([value.reshape(obs.shape[0], -1) for value in rows.values()], dim=1)


def ppo_direction(model, obs, act, adv, outputs_old):
    outputs = model.forward_pi(obs)
    ratio = torch.exp(selected_logp(outputs, act) - selected_logp(outputs_old, act))
    normalized_adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    losses = torch.maximum(-ratio * normalized_adv, -torch.clamp(ratio, 0.8, 1.2) * normalized_adv)
    grads = torch.autograd.grad(losses.mean(), tuple(model.pi_net.parameters()))
    return -flat_tensors(grads).detach()


def true_fisher_direction(model, obs, outputs_old, rhs, damping, cg_steps):
    params = tuple(model.pi_net.parameters())
    outputs = model.forward_pi(obs)
    mu, logstd = outputs.chunk(2, dim=-1)
    mu_old, logstd_old = outputs_old.chunk(2, dim=-1)
    kl = (
        logstd - logstd_old
        + 0.5 * (torch.exp(logstd_old).square() + (mu_old - mu).square()) / torch.exp(logstd).square()
        - 0.5
    ).sum(dim=-1).mean()
    kl_grad = torch.autograd.grad(kl, params, create_graph=True)
    kl_grad_flat = flat_tensors(kl_grad)

    def fvp(vector):
        product = torch.dot(kl_grad_flat, vector)
        hvp = torch.autograd.grad(product, params, retain_graph=True)
        return flat_tensors(hvp) + damping * vector

    return true_fisher_cg(fvp, rhs, cg_steps)


def kfac_direction(model, obs, act, adv, outputs_old, damping, scores=None,
                   precondition_shared_sigma=False):
    probe_model = copy.deepcopy(model.pi_net)
    # A one-shot same-minibatch comparison must use this batch's factors
    # directly; the training default (.95) otherwise mixes in identity state.
    optimizer = KFACOptimizer(
        probe_model,
        lr=1.0,
        damping=damping,
        stat_decay=0.0,
        kl_clip=float("inf") if precondition_shared_sigma else 0.001,
    )
    before = flat_tensors([p.detach().clone() for p in probe_model.parameters()])

    outputs = probe_model(obs)
    logp = selected_logp(outputs, act)
    with torch.no_grad():
        old_logp = selected_logp(outputs_old, act)
    ratio = torch.exp(logp - old_logp).clamp(0.1, 10.0)
    normalized_adv = (adv - adv.mean()) / (adv - adv.mean()).square().mean().sqrt().clamp_min(1e-8)

    optimizer.zero_grad()
    optimizer.acc_stats = True
    (-logp.mean()).backward(retain_graph=True)
    optimizer.acc_stats = False
    optimizer.zero_grad()
    (-(ratio * normalized_adv).mean()).backward()
    raw_update = -flat_tensors([
        parameter.grad.detach().clone() if parameter.grad is not None else torch.zeros_like(parameter)
        for parameter in probe_model.parameters()
    ])
    sigma_diagnostics = None
    if precondition_shared_sigma:
        if scores is None:
            raise ValueError("corrected K-FAC requires per-sample score rows")
        offset = 0
        sigma_parameter = None
        sigma_slice = None
        for name, parameter in probe_model.named_parameters():
            next_offset = offset + parameter.numel()
            if name == "shared_sigma":
                sigma_parameter = parameter
                sigma_slice = slice(offset, next_offset)
                break
            offset = next_offset
        if sigma_parameter is None or sigma_slice is None:
            raise RuntimeError("vector-sigma actor has no shared_sigma parameter")
        sigma_scores = scores[:, sigma_slice].to(dtype=torch.float64)
        sigma_fisher = sigma_scores.t().matmul(sigma_scores) / float(scores.shape[0])
        sigma_eye = torch.eye(
            sigma_fisher.shape[0], device=sigma_fisher.device, dtype=sigma_fisher.dtype
        )
        sigma_raw_grad = sigma_parameter.grad.detach().reshape(-1).to(dtype=torch.float64)
        sigma_natural_grad = torch.linalg.solve(
            sigma_fisher + damping * sigma_eye, sigma_raw_grad
        )
        sigma_parameter.grad.copy_(
            sigma_natural_grad.to(dtype=sigma_parameter.dtype).view_as(sigma_parameter)
        )
        sigma_diagnostics = {
            "parameter_slice": [sigma_slice.start, sigma_slice.stop],
            "raw_grad_norm": float(sigma_raw_grad.norm().item()),
            "natural_grad_norm": float(sigma_natural_grad.norm().item()),
            "fisher_trace": float(torch.trace(sigma_fisher).item()),
            "solve_dtype": "float64",
        }
    factor_stats = []
    for index, module in enumerate(optimizer.modules):
        factor_stats.append({
            "layer": index,
            "a_trace": float(torch.trace(optimizer.m_aa[module]).item()),
            "g_trace": float(torch.trace(optimizer.m_gg[module]).item()),
        })
    optimizer.step()
    after = flat_tensors([p.detach() for p in probe_model.parameters()])
    return after - before, raw_update, factor_stats, sigma_diagnostics


def render_heatmap(json_path):
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    matrix = np.asarray(payload["cosine"])
    fig, ax = plt.subplots(figsize=(8.2, 7.1))
    image = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    labels = payload["methods"]
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for row in range(len(labels)):
        for col in range(len(labels)):
            ax.text(col, row, f"{matrix[row, col]:.3f}", ha="center", va="center", fontsize=9)
    ax.set_title("MuJoCo MLP: same checkpoint and batch update cosine")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(Path(json_path).with_suffix(".png"), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="ant", choices=sorted(ENV_IDS))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup-updates", type=int, default=5)
    parser.add_argument("--minibatch", type=int, default=1024)
    parser.add_argument("--damping", type=float, default=0.03)
    parser.add_argument("--cg-steps", type=int, default=50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    set_seed(args.seed, torch_deterministic=True)
    device = torch.device(args.device)
    venv = make_vec_env(ENV_IDS[args.env], n_envs=32)
    obs_space = venv.observation_space
    build, preprocess = build_mlp(obs_space, device=device)
    action_dim = venv.action_space.shape[0]
    model = ActorCritic(
        build,
        obs_space.shape,
        nets_config=NetConfig(),
        dim_actions=action_dim,
        with_popart=True,
        sigma_type="vector",
        device=device,
    ).to(device)
    venv = VecNormalize(venv=venv, norm_ret=False, obs_preprocess=preprocess)
    runner = Runner(env=venv, model=model, nsteps=256, gamma=0.999, lam=0.95, adv_type="gae", device=device)

    ppo_warmup(model, runner, args.warmup_updates, args.minibatch)
    checkpoint = copy.deepcopy(model.state_dict())
    checkpoint_path = Path(args.output).with_suffix(".checkpoint.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": checkpoint, "warmup_updates": args.warmup_updates}, checkpoint_path)

    model.eval()
    obs, ret, act, adv, _, _ = runner.run()
    obs, _, adv = normalize_rollout(model, obs, ret, adv)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1000)
    chosen_cpu = torch.randperm(obs.shape[0], generator=generator)[:args.minibatch]
    chosen = chosen_cpu.to(device=obs.device)
    obs, act, adv = obs[chosen], act[chosen], adv[chosen]
    with torch.no_grad():
        outputs_old = model.forward_pi(obs).detach()
    ratios = torch.ones_like(adv)
    rat_adv = adv - adv.mean()
    rat_adv = rat_adv / rat_adv.square().mean().sqrt().clamp_min(1e-8)

    print("PROBE computing score rows", flush=True)
    scores = score_rows(model, obs, act).detach()
    rhs = scores.t().mv(ratios * rat_adv) / float(scores.shape[0])
    anchor_generator = torch.Generator(device="cpu").manual_seed(args.seed + 2000)
    anchors = torch.randperm(scores.shape[0], generator=anchor_generator)[:255].to(scores.device)
    emp_indices = torch.randperm(scores.shape[0], generator=anchor_generator)[:256].to(scores.device)

    directions = {}
    directions["ppo"] = ppo_direction(model, obs, act, adv, outputs_old)
    directions["emp256"] = empirical_direction(
        scores[emp_indices], rat_adv[emp_indices], ratios[emp_indices], args.damping
    )
    directions["emp256_fullrhs"] = empirical_subsample_fullrhs_direction(
        scores[emp_indices], rhs, ratios[emp_indices], args.damping
    )
    directions["energyfree255p1"], rho, energyfree_diagnostics = energyfree_255p1_direction(
        scores, rat_adv, ratios, anchors, args.damping
    )
    directions["full_empirical_direct"] = empirical_direction(
        scores, rat_adv, ratios, args.damping
    )
    full_direction = directions["full_empirical_direct"]
    full_fisher_direction = (
        scores.t().mv(ratios * scores.mv(full_direction)) / float(scores.shape[0])
        + args.damping * full_direction
    )
    full_residual = float(
        ((full_fisher_direction - rhs).norm() / rhs.norm().clamp_min(1e-12)).item()
    )
    directions["true_fisher_cg"], true_residual = true_fisher_direction(
        model, obs, outputs_old, rhs, args.damping, args.cg_steps
    )
    directions["kfac_hybrid"], kfac_raw_update, kfac_factor_stats, _ = kfac_direction(
        model, obs, act, adv, outputs_old, args.damping
    )
    directions["kfac_corrected"], _, kfac_corrected_factor_stats, kfac_sigma_diagnostics = kfac_direction(
        model, obs, act, adv, outputs_old, args.damping, scores=scores,
        precondition_shared_sigma=True
    )
    kfac_raw_vs_ppo = float(torch.nn.functional.cosine_similarity(
        kfac_raw_update.unsqueeze(0), directions["ppo"].unsqueeze(0)
    ).item())

    payload = {
        "definition": "six actor parameter-update directions from one PPO checkpoint and one fixed minibatch",
        "env": args.env,
        "seed": args.seed,
        "warmup_updates": args.warmup_updates,
        "warmup_transitions": args.warmup_updates * 8192,
        "batch_size": args.minibatch,
        "damping": args.damping,
        "cg_steps": args.cg_steps,
        "ratio_at_probe": 1.0,
        "emp256_indices": emp_indices.cpu().tolist(),
        "energyfree_anchor_indices": anchors.cpu().tolist(),
        "energyfree_rho": rho,
        "energyfree_diagnostics": energyfree_diagnostics,
        "full_empirical_direct_residual": full_residual,
        "true_fisher_cg_residual": true_residual,
        "kfac_config": {
            "damping": args.damping,
            "kl_clip": 0.001,
            "momentum": 0.9,
            "stat_decay": 0.0,
            "factor_source": "current fixed minibatch only",
        },
        "kfac_raw_update_vs_ppo_cosine": kfac_raw_vs_ppo,
        "kfac_factor_stats": kfac_factor_stats,
        "kfac_corrected_factor_stats": kfac_corrected_factor_stats,
        "kfac_sigma_diagnostics": kfac_sigma_diagnostics,
        "checkpoint": str(checkpoint_path),
    }
    save_probe(args.output, payload, directions)
    render_heatmap(args.output)
    print(f"PROBE_DONE output={args.output}", flush=True)


if __name__ == "__main__":
    main()
