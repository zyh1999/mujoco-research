#!/usr/bin/env python3
"""Plot archived MuJoCo learning curves from progress CSVs or trainer stdout."""

from __future__ import annotations

import csv
import glob
import math
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


REPO = "/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original"
RUNS = os.path.join(REPO, "perf_runs")
ARCHIVE = os.path.join(RUNS, "central_per_sample_records_20260710", "archive")
OUT = os.environ.get("MUJOCO_CURVE_OUT", "/tmp/mujoco_method_curves_20260719")

ENVS = [
    "ant",
    "halfcheetah",
    "hopper",
    "humanoid",
    "humanoidstandup",
    "swimmer",
    "walker2d",
]
ENV_TITLES = {
    "ant": "Ant",
    "halfcheetah": "HalfCheetah",
    "hopper": "Hopper",
    "humanoid": "Humanoid",
    "humanoidstandup": "HumanoidStandup",
    "swimmer": "Swimmer",
    "walker2d": "Walker2d",
}


def p(*parts: str) -> str:
    return os.path.join(*parts)


SHARED_EF_GGN_D03 = p(
    REPO,
    "logs",
    "shared.rat.karzmarz_False.mlp.a256x2x0.0e4x8.clip_grad_0.5.vector.damping_0.03.lr_0.05",
)
SHARED_EF_GGN_KTRUE = p(
    REPO,
    "logs",
    "shared.rat.karzmarz_True.mlp.a256x2x0.0e4x8.clip_grad_0.5.vector.damping_0.1.lr_0.03",
)
DETACH_EF_GGN_D03 = p(
    REPO,
    "logs",
    "rat.jointcritic.karzmarz_False.mlp.a256x2x0.0e4x8.c256x2x0.0e4x8.npg_fisher_clip_0.5.vector.damping_0.03.lr_pi_0.05.lr_v_0.1",
)
DETACH_EF_GGN_KTRUE = p(
    REPO,
    "logs",
    "rat.jointcritic.karzmarz_True.mlp.a256x2x0.0e4x8.c256x2x0.0e4x8.npg_fisher_clip_0.5.vector.damping_0.1.lr_pi_0.05.lr_v_0.1",
)
DETACH_EF_GGN_SWIMMER = p(
    REPO,
    "logs",
    "rat.jointcritic.karzmarz_True.mlp.a256x2x0.0e4x8.c256x2x0.0e4x8.npg_fisher_clip_0.5.vector.damping_0.1.lr_pi_0.05.lr_v_0.2",
)


METHODS = {
    "shared": {
        "PPO fixed lr (5s)": [
            p(ARCHIVE, "csf3", "public_ppo_fixedlr3e4_shared_detach_e4x8_20260716", "runs", "shared")
        ],
        "RAT EF+GGN tuned (5s)": [
            p(SHARED_EF_GGN_D03, f"{env}.*_[0-4]")
            for env in ("ant", "humanoid", "humanoidstandup", "swimmer", "walker2d")
        ] + [
            p(SHARED_EF_GGN_KTRUE, f"{env}.*_[0-4]")
            for env in ("halfcheetah", "hopper")
        ],
        "Exact-RAT d=.03 (5s)": [
            p(RUNS, "csf3_shared_exact_jointggn_normnone_damp003_kfalse_all7_5seed_10m_gpuA2_17491616")
        ],
        "Exact-RAT d=.1 (5s)": [
            p(RUNS, "csf3_shared_exact_jointggn_normnone_damp01_kfalse_all7_5seed_10m_gpuL2_17529603")
        ],
        "Curv256 d=.03 (5s)": [
            p(RUNS, "csf3_shared_curv256_ggn_normnone_damp003_kfalse_all7_5seed_10m_gpuL2_17505797")
        ],
        "Curv256 d=.1 (5s)": [
            p(RUNS, "csf3_shared_curv256_ggn_normnone_damp01_kfalse_all7_5seed_10m_gpuL2_17505023")
        ],
        "Energy-free255+1 d=.03 (5s)": [
            p(RUNS, "csf3_shared_energyfree255p1_ggn_normnone_damp003_kfalse_all7_5seed_10m_gpuL2_17538346")
        ],
        "Energy-free255+1 d=.1 (5s)": [
            p(RUNS, "csf3_shared_energyfree255p1_jointggn_normnone_damp01_hc_5seed_10m_gpuL2_17488041", "energyfree_kfalse"),
            p(RUNS, "csf3_shared_energyfree255p1_jointggn_normnone_damp01_remaining6_5seed_10m_gpuL2_17488191"),
        ],
        "Curv256 VF=4.0 (3s)": [
            p(RUNS, "csf3_shared_vfrhs_curv256_optuna_d003_kfalse_10m_3seed_20260713", "trials", "trial_006_vf_4p0")
        ],
        "Energy-free VF=1.5 (3s)": [
            p(RUNS, "csf3_shared_energyfree255p1_vfrhs_optuna_d003_kfalse_10m_3seed_20260715", "trials", "trial_004_vf_1p5")
        ],
        "Grouped-rho4 legacy d=.03 (5s)": [
            p(ARCHIVE, "groupedrho4_256dof_matrix_20260715", "shared_d003")
        ],
        "Grouped-rho4 legacy d=.1 (5s)": [
            p(ARCHIVE, "groupedrho4_256dof_matrix_20260715", "shared_d01")
        ],
        "Grouped-rho4 aligned d=.03 (5s)": [
            p(RUNS, "csf3_shared_groupedrho4_aligned1024_ggn_normnone_damp003_kfalse_all7_5seed_10m_gpuL2_17663499")
        ],
    },
    "no_shared": {
        "PPO fixed lr (5s)": [
            p(ARCHIVE, "csf3", "public_ppo_fixedlr3e4_shared_detach_e4x8_20260716", "runs", "detach")
        ],
        "RAT EF+GGN tuned (5s)": [
            p(DETACH_EF_GGN_D03, f"{env}.*_[0-4]")
            for env in ("ant", "humanoid", "walker2d")
        ] + [
            p(DETACH_EF_GGN_KTRUE, f"{env}.*_[0-4]")
            for env in ("halfcheetah", "hopper", "humanoidstandup")
        ] + [
            p(DETACH_EF_GGN_SWIMMER, "swimmer.*_[0-4]")
        ],
        "Exact-RAT d=.03 (5s)": [
            p(RUNS, "csf3_detjc_exact_ggn_normnone_damp003_kfalse_all7_5seed_10m_gpuL2_17491617")
        ],
        "Exact-RAT d=.1 (5s)": [
            p(RUNS, "csf3_detjc_exact_ggn_normnone_damp01_kfalse_all7_5seed_10m_gpuL2_17510355")
        ],
        "Curv256 d=.03 (5s)": [
            p(RUNS, "csf3_detjc_curv256_ggn_normnone_damp003_kfalse_all7_5seed_10m_gpuL2_17491618")
        ],
        "Curv256 d=.1 (5s)": [
            p(RUNS, "csf3_detjc_curv256_ggn_normnone_damp01_kfalse_all7_5seed_10m_gpuL2_17510356")
        ],
        "Energy-free255+1 d=.03 (5s)": [
            p(RUNS, "csf3_detach_energyfree255p1_ggn_normnone_damp003_kfalse_all7_5seed_10m_gpuL2_17538347")
        ],
        "Energy-free255+1 d=.1 (5s)": [
            p(RUNS, "csf3_detach_energyfree255p1_ggn_normnone_damp01_kfalse_all7_5seed_10m_gpuA2_17538348")
        ],
        "Grouped-rho4 legacy d=.03 (5s)": [
            p(ARCHIVE, "groupedrho4_256dof_matrix_20260715", "no_shared_d003")
        ],
        "Grouped-rho4 legacy d=.1 (5s)": [
            p(ARCHIVE, "groupedrho4_256dof_matrix_20260715", "no_shared_d01")
        ],
        "Grouped-rho4 aligned d=.03 (5s)": [
            p(ARCHIVE, "groupedrho4_aligned1024_20260717", "bede", "bede_detach_groupedrho4_aligned1024_ggn_normnone_damp003_kfalse_all7_5seed_10m_gpu2_1061987"),
            p(ARCHIVE, "groupedrho4_aligned1024_20260717", "bede", "bede_detach_groupedrho4_aligned1024_walker2d_ggn_normnone_damp003_kfalse_5seed_10m_gpu1_1062008"),
            p(ARCHIVE, "groupedrho4_aligned1024_20260717", "no_shared_swimmer_repair", "run_root"),
        ],
        "Hier16x64 d=.03 (5s)": [
            p(RUNS, "csf3_detach_hier16x64_aligned1024_ggn_normnone_damp003_kfalse_ant_5seed_10m_gpuL1_17689401_0"),
            p(RUNS, "csf3_detach_hier16x64_aligned1024_ggn_normnone_damp003_kfalse_halfcheetah_5seed_10m_gpuL1_17688137"),
            p(RUNS, "csf3_detach_hier16x64_aligned1024_ggn_normnone_damp003_kfalse_hopper_5seed_10m_gpuL1_17689401_1"),
            p(RUNS, "csf3_detach_hier16x64_aligned1024_ggn_normnone_damp003_kfalse_humanoid_5seed_10m_gpuL1_17689562_2"),
            p(RUNS, "csf3_detach_hier16x64_aligned1024_ggn_normnone_damp003_kfalse_humanoidstandup_5seed_10m_gpuL1_17689562_3"),
            p(RUNS, "csf3_detach_hier16x64_aligned1024_ggn_normnone_damp003_kfalse_swimmer_5seed_10m_gpuL1_17689401_4"),
            p(RUNS, "csf3_detach_hier16x64_aligned1024_ggn_normnone_damp003_kfalse_walker2d_5seed_10m_gpuL1_17689401_5"),
        ],
    },
}

CORE = [
    "PPO fixed lr (5s)",
    "RAT EF+GGN tuned (5s)",
    "Curv256 d=.03 (5s)",
    "Energy-free255+1 d=.03 (5s)",
]


def infer_env(path: str) -> str | None:
    low = path.lower()
    if re.search(r"(^|[/_.-])hc([/_.-]|$)", low):
        return "halfcheetah"
    for env in sorted(ENVS, key=len, reverse=True):
        if re.search(rf"(^|[/_.-]){env}([/_.-]|$)", low):
            return env
    return None


def infer_seed(path: str) -> int | None:
    for pattern in (r"seed([0-4])_work", r"seed([0-4])(?:\.|/|$)", r"_([0-4])/progress\.csv$"):
        match = re.search(pattern, path)
        if match:
            return int(match.group(1))
    return None


def read_csv_curve(path: str) -> tuple[np.ndarray, np.ndarray]:
    steps, rewards = [], []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                step = float(row.get("misc/total_timesteps", row.get("total_timesteps", "nan")))
                reward = float(row["eprewmean"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(step) and math.isfinite(reward):
                steps.append(step)
                rewards.append(reward)
    return np.asarray(steps), np.asarray(rewards)


def read_stdout_curve(path: str) -> tuple[np.ndarray, np.ndarray]:
    steps, rewards = [], []
    current_reward = None
    value_pattern = re.compile(r"\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|")
    with open(path, errors="replace") as handle:
        for line in handle:
            match = value_pattern.search(line)
            if not match:
                continue
            key, value = match.group(1).strip(), match.group(2).strip()
            try:
                number = float(value)
            except ValueError:
                continue
            if key == "eprewmean":
                current_reward = number
            elif key == "misc/total_timesteps" and current_reward is not None:
                steps.append(number)
                rewards.append(current_reward)
                current_reward = None
    return np.asarray(steps), np.asarray(rewards)


def smooth(values: np.ndarray, window: int = 10) -> np.ndarray:
    if values.size == 0:
        return values
    out = np.empty_like(values, dtype=float)
    csum = np.cumsum(np.insert(values.astype(float), 0, 0.0))
    for index in range(values.size):
        start = max(0, index - window + 1)
        out[index] = (csum[index + 1] - csum[start]) / (index - start + 1)
    return out


def collect_method(sources: list[str]) -> dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]:
    candidates: dict[tuple[str, int], list[tuple[np.ndarray, np.ndarray, float]]] = defaultdict(list)
    for source in sources:
        for path in glob.glob(p(source, "**", "progress.csv"), recursive=True):
            if "preflight" in path:
                continue
            env, seed = infer_env(path), infer_seed(path)
            if env is None or seed is None:
                continue
            curve = read_csv_curve(path)
            if curve[0].size:
                candidates[(env, seed)].append((*curve, os.path.getmtime(path)))
        for path in glob.glob(p(source, "**", "seed*.stdout"), recursive=True):
            if "preflight" in path:
                continue
            env, seed = infer_env(path), infer_seed(path)
            if env is None or seed is None:
                continue
            curve = read_stdout_curve(path)
            if curve[0].size:
                candidates[(env, seed)].append((*curve, os.path.getmtime(path)))

    result: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = defaultdict(dict)
    for (env, seed), curves in candidates.items():
        steps, rewards, _ = max(curves, key=lambda item: (item[0][-1], item[2]))
        order = np.argsort(steps)
        steps, rewards = steps[order], rewards[order]
        unique_steps, unique_indices = np.unique(steps, return_index=True)
        result[env][seed] = (unique_steps, smooth(rewards[unique_indices]))
    return result


def aggregate(curves: dict[int, tuple[np.ndarray, np.ndarray]], grid: np.ndarray):
    rows = []
    for steps, rewards in curves.values():
        row = np.interp(grid, steps, rewards)
        row[(grid < steps[0]) | (grid > steps[-1])] = np.nan
        rows.append(row)
    if not rows:
        return None
    matrix = np.vstack(rows)
    count = np.sum(np.isfinite(matrix), axis=0)
    mean = np.nanmean(matrix, axis=0)
    stderr = np.nanstd(matrix, axis=0, ddof=1) / np.sqrt(np.maximum(count, 1))
    mean[count < 2] = np.nan
    stderr[count < 2] = np.nan
    return mean, stderr, count


def plot_group(group: str, names: list[str], data, filename: str, title: str):
    grid = np.arange(0.1e6, 10.0001e6, 0.1e6)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.7), constrained_layout=True)
    axes = axes.ravel()
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, min(10, len(names))))
    line_styles = ["-", "--", "-.", ":"]
    handles = []
    labels = []
    for env_index, env in enumerate(ENVS):
        ax = axes[env_index]
        for method_index, name in enumerate(names):
            aggregated = aggregate(data[group][name].get(env, {}), grid)
            if aggregated is None:
                continue
            mean, stderr, count = aggregated
            valid = np.isfinite(mean)
            if not np.any(valid):
                continue
            color = colors[method_index % len(colors)]
            style = line_styles[(method_index // len(colors)) % len(line_styles)]
            line, = ax.plot(grid[valid] / 1e6, mean[valid], color=color, linestyle=style, linewidth=1.65, label=name)
            ax.fill_between(grid[valid] / 1e6, mean[valid] - stderr[valid], mean[valid] + stderr[valid], color=color, alpha=0.10, linewidth=0)
            if env_index == 0:
                handles.append(line)
                labels.append(name)
        ax.set_title(ENV_TITLES[env], fontsize=12)
        ax.grid(True, alpha=0.22, linewidth=0.7)
        ax.set_xlim(0, 10)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
        if env_index >= 4:
            ax.set_xlabel("Environment steps (M)")
        if env_index % 4 == 0:
            ax.set_ylabel("Episode return")
    axes[-1].axis("off")
    axes[-1].legend(handles, labels, loc="center left", frameon=False, fontsize=9.5)
    fig.suptitle(title + " | per-seed rolling-10, mean ± stderr", fontsize=15)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(p(OUT, filename), dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_inventory(data):
    with open(p(OUT, "curve_inventory.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "method", "environment", "seeds", "min_final_steps", "max_final_steps"])
        for group, methods in data.items():
            for name, envs in methods.items():
                for env in ENVS:
                    curves = envs.get(env, {})
                    finals = [curve[0][-1] for curve in curves.values() if curve[0].size]
                    writer.writerow([group, name, env, len(finals), min(finals) if finals else "", max(finals) if finals else ""])


def write_final_summary(data):
    with open(p(OUT, "final_last10_summary.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "method", "environment", "seeds", "mean_last10", "stderr_last10"])
        for group, methods in data.items():
            for name, envs in methods.items():
                for env in ENVS:
                    values = [curve[1][-1] for curve in envs.get(env, {}).values() if curve[1].size]
                    if not values:
                        continue
                    stderr = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else float("nan")
                    writer.writerow([group, name, env, len(values), np.mean(values), stderr])


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {group: {} for group in METHODS}
    for group, methods in METHODS.items():
        for name, sources in methods.items():
            data[group][name] = collect_method(sources)

    write_inventory(data)
    write_final_summary(data)
    shared_variants = [name for name in METHODS["shared"] if name not in CORE]
    no_shared_variants = [name for name in METHODS["no_shared"] if name not in CORE]
    plot_group("shared", CORE, data, "shared_core.png", "Shared MuJoCo: PPO, tuned RAT EF+GGN, Curv256, Energy-free")
    plot_group("no_shared", CORE, data, "no_shared_core.png", "No-shared MuJoCo: PPO, tuned RAT EF+GGN, Curv256, Energy-free")
    plot_group("shared", shared_variants, data, "shared_variants.png", "Shared MuJoCo: damping, VF-RHS, and grouped 256-DOF variants")
    plot_group("no_shared", no_shared_variants, data, "no_shared_variants.png", "No-shared MuJoCo: damping, grouped, aligned, and hierarchical variants")
    print(OUT)


if __name__ == "__main__":
    main()
