#!/usr/bin/env python3
"""Export audited MuJoCo endpoints and uniformly gridded learning curves.

This script is intended to run read-only inside the CSF3 repository and emit
one compact JSON document on stdout.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from pathlib import Path


REPO = Path("/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original")
RUNS = REPO / "perf_runs"
ARCHIVE = RUNS / "central_per_sample_records_20260710" / "archive"
ENVS = [
    "ant",
    "halfcheetah",
    "hopper",
    "humanoid",
    "humanoidstandup",
    "swimmer",
    "walker2d",
]
GRID = [float(x) for x in range(0, 10_000_001, 500_000)]
VALUE_RE = re.compile(r"\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|")
ERROR_RE = re.compile(r"OOM|out of memory|NaN|Traceback|LinAlgError", re.I)
LOG_PATH_RE = re.compile(r"^Logging to\s+(\S+)\s*$")


PPO_ARCHIVE = (
    ARCHIVE / "csf3" / "public_ppo_fixedlr3e4_shared_detach_e4x8_20260716"
)
EXACT_SHARED = RUNS / "csf3_shared_exact_jointggn_normnone_damp003_kfalse_all7_5seed_10m_gpuA2_17491616"
EXACT_DETACH = RUNS / "csf3_detjc_exact_ggn_normnone_damp003_kfalse_all7_5seed_10m_gpuL2_17491617"
EXACT_SHARED_D01 = RUNS / "csf3_shared_exact_jointggn_normnone_damp01_kfalse_all7_5seed_10m_gpuL2_17529603"
EXACT_DETACH_D01 = RUNS / "csf3_detjc_exact_ggn_normnone_damp01_kfalse_all7_5seed_10m_gpuL2_17510355"
CURV_SHARED_VF = RUNS / "csf3_shared_vfrhs_curv256_optuna_d003_kfalse_10m_3seed_20260713"
CURV_SHARED_VF4 = CURV_SHARED_VF / "trials" / "trial_006_vf_4p0"
CURV_DETACH_D01 = RUNS / "csf3_detjc_curv256_ggn_normnone_damp01_kfalse_all7_5seed_10m_gpuL2_17510356"
CURV_DETACH_D003 = RUNS / "csf3_detjc_curv256_ggn_normnone_damp003_kfalse_all7_5seed_10m_gpuL2_17491618"


def finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_progress(path: Path) -> tuple[list[float], list[float]]:
    points: dict[float, float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            step = finite_float(row.get("misc/total_timesteps", row.get("total_timesteps")))
            reward = finite_float(row.get("eprewmean"))
            if step is not None and reward is not None:
                points[step] = reward
    steps = sorted(points)
    return steps, [points[step] for step in steps]


def read_stdout(path: Path) -> tuple[list[float], list[float]]:
    points: dict[float, float] = {}
    reward: float | None = None
    with path.open(errors="replace") as handle:
        for line in handle:
            match = VALUE_RE.search(line)
            if not match:
                continue
            key, raw = match.group(1).strip(), match.group(2).strip()
            value = finite_float(raw)
            if value is None:
                continue
            if key == "eprewmean":
                reward = value
            elif key == "misc/total_timesteps" and reward is not None:
                points[value] = reward
                reward = None
    steps = sorted(points)
    return steps, [points[step] for step in steps]


def read_curve(path: Path, fmt: str) -> tuple[list[float], list[float]]:
    if fmt == "progress":
        return read_progress(path)
    with path.open(errors="replace") as handle:
        for line in handle:
            match = LOG_PATH_RE.match(line.strip())
            if match:
                linked = REPO / match.group(1) / "progress.csv"
                if linked.exists():
                    return read_progress(linked)
                break
    return read_stdout(path)


def rolling_mean(values: list[float], window: int = 10) -> list[float]:
    result = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        chunk = values[start : index + 1]
        result.append(sum(chunk) / len(chunk))
    return result


def interpolate(steps: list[float], values: list[float], grid: list[float]) -> list[float]:
    if not steps:
        raise ValueError("cannot interpolate an empty curve")
    result = []
    right = 0
    for target in grid:
        if target <= steps[0]:
            result.append(values[0])
            continue
        if target >= steps[-1]:
            result.append(values[-1])
            continue
        while right + 1 < len(steps) and steps[right + 1] < target:
            right += 1
        left_step, right_step = steps[right], steps[right + 1]
        left_value, right_value = values[right], values[right + 1]
        weight = (target - left_step) / (right_step - left_step)
        result.append(left_value + weight * (right_value - left_value))
    return result


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def progress_path(root: Path, mode: str, env: str, seed: int) -> Path:
    matches = sorted((root / "runs" / mode / env / f"seed{seed}_work").glob("**/progress.csv"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one progress.csv for {mode}/{env}/seed{seed}, got {matches}")
    return matches[0]


def stdout_path(root: Path, env: str, seed: int) -> Path:
    return root / env / f"seed{seed}.stdout"


def status_text(path: Path) -> str | None:
    return path.read_text().strip() if path.exists() else None


def cell_sources(spec: dict, env: str) -> list[Path]:
    paths = []
    for seed in spec["seeds"]:
        if spec["format"] == "progress":
            paths.append(progress_path(spec["root"], spec["mode"], env, seed))
        else:
            paths.append(stdout_path(spec["root"], env, seed))
    return paths


def scan_errors(spec: dict, env: str) -> list[str]:
    hits = []
    if spec["format"] == "progress":
        base = spec["root"] / "runs" / spec["mode"] / env
    else:
        base = spec["root"] / env
    for path in sorted(base.glob("seed*.stderr")):
        with path.open(errors="replace") as handle:
            for line_number, line in enumerate(handle, 1):
                if ERROR_RE.search(line):
                    hits.append(f"{path}:{line_number}:{line.strip()[:160]}")
    return hits


def collect_cell(architecture: str, method: str, spec: dict, env: str) -> tuple[dict, dict]:
    seed_curves = []
    endpoint_values = []
    final_steps = []
    source_paths = cell_sources(spec, env)
    for seed, path in zip(spec["seeds"], source_paths):
        steps, raw_rewards = read_curve(path, spec["format"])
        if not steps:
            raise RuntimeError(f"empty curve: {path}")
        smoothed = rolling_mean(raw_rewards, 10)
        endpoint_values.append(smoothed[-1])
        final_steps.append(steps[-1])
        seed_curves.append(interpolate(steps, smoothed, GRID))

    count = len(seed_curves)
    curve_mean = [mean([curve[index] for curve in seed_curves]) for index in range(len(GRID))]
    curve_min = [min(curve[index] for curve in seed_curves) for index in range(len(GRID))]
    curve_max = [max(curve[index] for curve in seed_curves) for index in range(len(GRID))]
    table = {
        "architecture": architecture,
        "method": method,
        "environment": env,
        "availability": spec["availability"],
        "seed_count": count,
        "seeds": spec["seeds"],
        "nominal_horizon": 10_000_000,
        "min_final_step": min(final_steps),
        "max_final_step": max(final_steps),
        "endpoint_last10_mean": mean(endpoint_values),
        "endpoint_last10_std": sample_std(endpoint_values),
        "endpoint_last10_min": min(endpoint_values),
        "endpoint_last10_max": max(endpoint_values),
        "seed_endpoint_last10": endpoint_values,
    }
    curve = {
        "architecture": architecture,
        "method": method,
        "environment": env,
        "metric": "per-seed rolling-last-10 eprewmean, linearly interpolated",
        "grid_steps": GRID,
        "mean": curve_mean,
        "min": curve_min,
        "max": curve_max,
    }
    return table, curve


def endpoint_means(root: Path, seeds: list[int], fmt: str = "stdout", mode: str | None = None) -> dict[str, float]:
    result = {}
    spec = {"root": root, "seeds": seeds, "format": fmt, "mode": mode}
    for env in ENVS:
        values = []
        for path in cell_sources(spec, env):
            steps, rewards = read_curve(path, fmt)
            if steps[-1] < 9_500_000:
                raise RuntimeError(f"incomplete candidate curve: {path} at {steps[-1]}")
            values.append(rolling_mean(rewards, 10)[-1])
        result[env] = mean(values)
    return result


def normalized_score(candidate: dict[str, float], reference: dict[str, float]) -> float:
    return mean([candidate[env] / reference[env] for env in ENVS])


def main() -> None:
    specs = {
        ("shared", "ppo_fixed_lr"): {
            "root": PPO_ARCHIVE,
            "mode": "shared",
            "format": "progress",
            "seeds": [0, 1, 2, 3, 4],
            "availability": "complete",
        },
        ("no_shared", "ppo_fixed_lr"): {
            "root": PPO_ARCHIVE,
            "mode": "detach",
            "format": "progress",
            "seeds": [0, 1, 2, 3, 4],
            "availability": "complete",
        },
        ("shared", "exact_ef_ggn"): {
            "root": EXACT_SHARED,
            "format": "stdout",
            "seeds": [0, 1, 2, 3, 4],
            "availability": "complete",
        },
        ("no_shared", "exact_ef_ggn"): {
            "root": EXACT_DETACH,
            "format": "stdout",
            "seeds": [0, 1, 2, 3, 4],
            "availability": "complete",
        },
        ("shared", "ordinary_curv256_ggn256"): {
            "root": CURV_SHARED_VF4,
            "format": "stdout",
            "seeds": [0, 1, 2],
            "availability": "partial_3seed_full7env_10m",
        },
        ("no_shared", "ordinary_curv256_ggn256"): {
            "root": CURV_DETACH_D003,
            "format": "stdout",
            "seeds": [0, 1, 2, 3, 4],
            "availability": "complete",
        },
    }

    table, curves, validation = [], [], []
    for (architecture, method), spec in specs.items():
        for env in ENVS:
            row, curve = collect_cell(architecture, method, spec, env)
            table.append(row)
            curves.append(curve)
            if spec["format"] == "progress":
                status = status_text(spec["root"] / "runs" / spec["mode"] / env / "status")
                rc_values = [
                    status_text(spec["root"] / "runs" / spec["mode"] / env / f"seed{seed}.rc")
                    for seed in spec["seeds"]
                ]
            else:
                status = status_text(spec["root"] / env / "status")
                rc_values = [status_text(spec["root"] / env / f"seed{seed}.rc") for seed in spec["seeds"]]
            validation.append({
                "architecture": architecture,
                "method": method,
                "environment": env,
                "status": status,
                "seed_rc": rc_values,
                "error_hits": scan_errors(spec, env),
            })

    exact_shared_ref = endpoint_means(EXACT_SHARED, [0, 1, 2, 3, 4])
    exact_shared_d01 = endpoint_means(EXACT_SHARED_D01, [0, 1, 2, 3, 4])
    exact_detach_ref = endpoint_means(EXACT_DETACH, [0, 1, 2, 3, 4])
    exact_detach_d01 = endpoint_means(EXACT_DETACH_D01, [0, 1, 2, 3, 4])
    curv_detach_ref = endpoint_means(CURV_DETACH_D003, [0, 1, 2, 3, 4])
    curv_detach_d01 = endpoint_means(CURV_DETACH_D01, [0, 1, 2, 3, 4])

    vf_candidates = []
    for trial in sorted((CURV_SHARED_VF / "trials").glob("trial_*")):
        if status_text(trial / "status") != "FINISHED":
            continue
        info = dict(
            line.split("=", 1) for line in (trial / "trial_info.txt").read_text().splitlines() if "=" in line
        )
        vf_candidates.append({
            "trial": trial.name,
            "vf_coef": float(info["vf_coef"]),
            "normalized_objective": float((trial / "objective.txt").read_text()),
            "coverage": "7env_x_3seed_x_10m",
        })

    configs = {
        "ppo_fixed_lr": {
            "shared": {
                "update": "public PPO_Update; shared network and single Adam",
                "actor_epochs_minibatches": "4x8",
                "critic_epochs_minibatches": "4x8",
                "lr": 0.0003,
                "lr_schedule": "fixed; use_kl_adaptive_lr=false",
                "cliprange": 0.2,
                "vf_coef": 1.0,
                "popart": True,
                "normalization": "norm_obj=adv; norm_obs=true; norm_ret=false",
                "rollout_minibatch": "32 envs x 256 steps = 8192; minibatch=1024",
            },
            "no_shared": {
                "update": "public PPO_Update; independent actor/critic and two Adam optimizers",
                "actor_epochs_minibatches": "4x8",
                "critic_epochs_minibatches": "4x8",
                "lr_pi": 0.0003,
                "lr_v": 0.0003,
                "lr_schedule": "fixed; use_kl_adaptive_lr=false",
                "cliprange": 0.2,
                "popart": True,
                "normalization": "norm_obj=adv; norm_obs=true; norm_ret=false",
                "rollout_minibatch": "32 envs x 256 steps = 8192; minibatch=1024",
            },
        },
        "exact_ef_ggn": {
            "shared": {
                "actor": "exact empirical-Fisher sample-space RAT over all 1024 minibatch samples",
                "critic": "per-sample GGN jointly solved over all 1024 critic rows",
                "actor_epochs_minibatches": "4x8",
                "critic_epochs_minibatches": "joint 4x8",
                "optimizer": "SGD semantics",
                "lr": 0.05,
                "lr_schedule": "fixed in executed path; config flag use_kl_adaptive_lr=true is inert here",
                "damping": 0.03,
                "kaczmarz": False,
                "solve": "dense torch.linalg.solve",
                "ratio_inside_inverse": True,
                "vf_coef": 1.0,
                "clip_normalization": "ratio clamp [0.1,10], grad clip 0.5, fisher normalization none",
                "popart": True,
            },
            "no_shared": {
                "actor": "exact empirical-Fisher sample-space RAT over all 1024 minibatch samples",
                "critic": "independent per-sample GGN over all 1024 critic rows",
                "actor_epochs_minibatches": "4x8",
                "critic_epochs_minibatches": "4x8",
                "optimizer": "actor SGD; critic SGD semantics",
                "lr_pi": 0.05,
                "lr_v": 0.1,
                "lr_schedule": "fixed in executed path; config flag use_kl_adaptive_lr=true is inert here",
                "damping": 0.03,
                "kaczmarz": False,
                "solve": "solve_score_kernel_system, exact kernel",
                "ratio_inside_inverse": True,
                "critic_coef": "independent critic residual; no shared vf coupling coefficient",
                "clip_normalization": "ratio clamp [0.1,10], actor grad clip 0.5, critic grad clip 5, fisher normalization none",
                "popart": True,
            },
        },
        "ordinary_curv256_ggn256": {
            "shared": {
                "actor": "ordinary 256-anchor empirical-Fisher solve; anchor-only RHS/update, not full-gradient/free-rho",
                "critic": "256-row GGN using the same sampled indices",
                "actor_epochs_minibatches": "4x8",
                "critic_epochs_minibatches": "joint 4x8",
                "optimizer": "SGD semantics",
                "lr": 0.05,
                "lr_schedule": "fixed in executed path",
                "damping": 0.03,
                "kaczmarz": False,
                "solve": "dense exact sample-kernel solve",
                "ratio_inside_inverse": True,
                "vf_coef": 4.0,
                "vf_mode": "rhs_only; joint curvature metric unchanged",
                "clip_normalization": "ratio clamp [0.1,10], grad clip 0.5, fisher normalization none",
                "popart": True,
            },
            "no_shared": {
                "actor": "ordinary 256-anchor empirical-Fisher solve; anchor-only RHS/update, not full-gradient/free-rho",
                "critic": "independent 256-row GGN",
                "actor_epochs_minibatches": "4x8",
                "critic_epochs_minibatches": "4x8",
                "optimizer": "actor SGD; critic SGD semantics",
                "lr_pi": 0.05,
                "lr_v": 0.1,
                "lr_schedule": "fixed in executed path",
                "damping": 0.03,
                "kaczmarz": False,
                "solve": "solve_score_kernel_system, exact 256-row kernel",
                "ratio_inside_inverse": True,
                "critic_coef": "independent critic residual",
                "clip_normalization": "ratio clamp [0.1,10], actor grad clip 0.5, critic grad clip 5, fisher normalization none",
                "popart": True,
            },
        },
    }

    document = {
        "schema_version": 1,
        "generated_utc": os.environ.get("AUDIT_GENERATED_UTC", "2026-07-19"),
        "environments": ENVS,
        "curve_grid_rule": "0..10M every 0.5M; per-seed rolling-last-10 eprewmean; linear interpolation; endpoint hold outside observed range",
        "provenance": {
            "repo": str(REPO),
            "ppo_archive": str(PPO_ARCHIVE),
            "exact_shared": str(EXACT_SHARED),
            "exact_no_shared": str(EXACT_DETACH),
            "curv256_shared_vf_sweep": str(CURV_SHARED_VF),
            "curv256_shared_selected": str(CURV_SHARED_VF4),
            "curv256_no_shared_selected": str(CURV_DETACH_D003),
            "verified_inputs": [
                "run_info/manifest/status/seed rc where present",
                "archived PPO README, validation.tsv, source manifests, configs",
                "executed stdout/progress curves",
                "trainer source ratio-weighting and subsample branches",
            ],
            "artifacts": {
                "ppo_fixed_lr": {
                    "archive_readme": str(PPO_ARCHIVE / "README.md"),
                    "shared_source_job": "17595177",
                    "no_shared_source_job": "17595178",
                    "shared_trainer": str(REPO / "train_shared_ppo_public.py"),
                    "no_shared_trainer": str(REPO / "train_detach_ppo_public.py"),
                    "shared_config": str(REPO / "launchers/ppo_mlp_shared_fixedlr_public.yaml"),
                    "no_shared_config": str(REPO / "launchers/ppo_mlp_detach_fixedlr_public.yaml"),
                    "env_launcher": str(REPO / "launchers/launch_public_ppo_fixedlr_env.sh"),
                    "shared_slurm_launcher": str(REPO / "launchers/sbatch_shared_ppo_fixedlr_public_gpuL_array.sh"),
                    "no_shared_slurm_launcher": str(REPO / "launchers/sbatch_detach_ppo_fixedlr_public_gpuL_array.sh"),
                    "command_template": "python -u <shared_or_detach_public_trainer> --config <matching_fixedlr_public_yaml> --env_name ENV --seed SEED --device GPU --timesteps_per_proc 10000000",
                },
                "exact_ef_ggn": {
                    "shared_run_info": str(EXACT_SHARED / "run_info.txt"),
                    "no_shared_run_info": str(EXACT_DETACH / "run_info.txt"),
                    "shared_trainer": str(REPO / "train_shared_jointkernel.py"),
                    "no_shared_trainer": str(REPO / "train_detach_jointcritic.py"),
                    "shared_config": str(REPO / "configs/rat_mlp_shared.yaml"),
                    "no_shared_config": str(REPO / "configs/rat_mlp_detjc_normnone_kfalse_d003_lrv01_e4_mlp.yaml"),
                    "shared_command_template": "python -u train_shared_jointkernel.py --config configs/rat_mlp_shared.yaml --env_name ENV --seed SEED --timesteps_per_proc 10000000 --epochs 4 --lr 0.05 --cg_damping 0.03 --fisher_kernel exact --fisher_kernel_normalization none --actor_curvature_subsample 0 --critic_curvature_subsample 0 --no_karzmarz",
                    "no_shared_command_template": "python -u train_detach_jointcritic.py --config configs/rat_mlp_detjc_normnone_kfalse_d003_lrv01_e4_mlp.yaml --env_name ENV --seed SEED --timesteps_per_proc 10000000 --epochs 4 --lr_pi 0.05 --lr_v 0.1 --cg_damping 0.03 --fisher_kernel exact --fisher_kernel_normalization none --actor_curvature_subsample 0 --critic_curvature_subsample 0 --no_karzmarz",
                },
                "ordinary_curv256_ggn256": {
                    "shared_sweep_run_info": str(CURV_SHARED_VF / "run_info.txt"),
                    "shared_trial_info": str(CURV_SHARED_VF4 / "trial_info.txt"),
                    "shared_trial_objective": str(CURV_SHARED_VF4 / "objective.txt"),
                    "shared_trainer": str(REPO / "train_shared_jointkernel_vfrhs.py"),
                    "no_shared_run_info": str(CURV_DETACH_D003 / "run_info.txt"),
                    "no_shared_trainer": str(REPO / "train_detach_jointcritic.py"),
                    "shared_command_template": "python -u train_shared_jointkernel_vfrhs.py --config configs/rat_mlp_shared.yaml --env_name ENV --seed SEED --timesteps_per_proc 10000000 --epochs 4 --lr 0.05 --vf_coef 4.0 --cg_damping 0.03 --fisher_kernel exact --fisher_kernel_normalization none --actor_curvature_subsample 256 --critic_curvature_subsample 256 --no_karzmarz",
                    "no_shared_command_template": "python -u train_detach_jointcritic.py --config configs/rat_mlp_detjc_normnone_kfalse_d003_lrv01_e4_mlp.yaml --env_name ENV --seed SEED --timesteps_per_proc 10000000 --epochs 4 --lr_pi 0.05 --lr_v 0.1 --cg_damping 0.03 --fisher_kernel exact --fisher_kernel_normalization none --actor_curvature_subsample 256 --critic_curvature_subsample 256 --no_karzmarz",
                },
            },
        },
        "selection": {
            "criterion": "Among formal 7-environment 10M groups, maximize the arithmetic mean of per-environment endpoint-last10 ratios to the designated same-family reference; partial seed count is retained and labeled.",
            "exact_shared": {
                "selected": "d=.03,K=false,lr=.05",
                "reference": "d=.03",
                "candidates": [
                    {"name": "d=.03,K=false", "coverage": "7env_x_5seed_x_10m", "normalized_score": 1.0},
                    {"name": "d=.1,K=false", "coverage": "7env_x_5seed_x_10m", "normalized_score": normalized_score(exact_shared_d01, exact_shared_ref)},
                ],
                "excluded": "K=true candidates lacked complete seven-environment coverage",
            },
            "exact_no_shared": {
                "selected": "d=.03,K=false,lr_pi=.05,lr_v=.1",
                "reference": "d=.03",
                "candidates": [
                    {"name": "d=.03,K=false", "coverage": "7env_x_5seed_x_10m", "normalized_score": 1.0},
                    {"name": "d=.1,K=false", "coverage": "7env_x_5seed_x_10m", "normalized_score": normalized_score(exact_detach_d01, exact_detach_ref)},
                ],
                "excluded": "K=true candidates lacked complete seven-environment coverage",
            },
            "curv256_shared": {
                "selected": "vf=4.0 rhs_only,d=.03,K=false,lr=.05",
                "candidates": vf_candidates,
                "warning": "Best full-seven-environment formal VF sweep has only 3 seeds, so this cell is partial relative to 5-seed cells.",
            },
            "curv256_no_shared": {
                "selected": "d=.03,K=false,lr_pi=.05,lr_v=.1",
                "reference": "d=.03",
                "candidates": [
                    {"name": "d=.03,K=false", "coverage": "7env_x_5seed_x_10m", "normalized_score": 1.0},
                    {"name": "d=.1,K=false", "coverage": "7env_x_5seed_x_10m", "normalized_score": normalized_score(curv_detach_d01, curv_detach_ref)},
                ],
                "excluded": "K=true/lr_v=.001 candidates lacked complete seven-environment coverage",
            },
        },
        "configs": configs,
        "validation": validation,
        "table": table,
        "curves": curves,
    }
    print(json.dumps(document, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
