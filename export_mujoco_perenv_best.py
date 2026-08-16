#!/usr/bin/env python3
"""Export per-environment best audited MuJoCo PPO/RAT/curv256 curves."""

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
PPO = ARCHIVE / "csf3" / "public_ppo_fixedlr3e4_shared_detach_e4x8_20260716"
ENVS = ["ant", "halfcheetah", "hopper", "humanoid", "humanoidstandup", "swimmer", "walker2d"]
GRID = [float(x) for x in range(0, 10_000_001, 500_000)]
LOG_RE = re.compile(r"^Logging to\s+(\S+)\s*$")
ERROR_RE = re.compile(r"OOM|out of memory|NaN|Traceback|LinAlgError", re.I)

EX_S_D003 = RUNS / "csf3_shared_exact_jointggn_normnone_damp003_kfalse_all7_5seed_10m_gpuA2_17491616"
EX_N_D003 = RUNS / "csf3_detjc_exact_ggn_normnone_damp003_kfalse_all7_5seed_10m_gpuL2_17491617"
EX_S_D01 = RUNS / "csf3_shared_exact_jointggn_normnone_damp01_kfalse_all7_5seed_10m_gpuL2_17529603"
EX_N_D01 = RUNS / "csf3_detjc_exact_ggn_normnone_damp01_kfalse_all7_5seed_10m_gpuL2_17510355"
CU_S_D003 = RUNS / "csf3_shared_curv256_ggn_normnone_damp003_kfalse_all7_5seed_10m_gpuL2_17505797"
CU_S_D01 = RUNS / "csf3_shared_curv256_ggn_normnone_damp01_kfalse_all7_5seed_10m_gpuL2_17505023"
CU_N_D003 = RUNS / "csf3_detjc_curv256_ggn_normnone_damp003_kfalse_all7_5seed_10m_gpuL2_17491618"
CU_N_D01 = RUNS / "csf3_detjc_curv256_ggn_normnone_damp01_kfalse_all7_5seed_10m_gpuL2_17510356"
RHS_ROOT = RUNS / "csf3_shared_vfrhs_curv256_optuna_d003_kfalse_10m_3seed_20260713" / "trials"
EXACT_VF_SCREEN = RUNS / "csf3_hopper_shared_jointggn_normnone_vf_screen_8x3seed_6m_jupyter17404349_20260710"

LOGS = REPO / "logs"
S_EX_KT_LR03 = LOGS / "shared.rat.karzmarz_True.mlp.a256x2x0.0e4x8.clip_grad_0.5.vector.damping_0.1.lr_0.03"
N_EX_KT = LOGS / "rat.jointcritic.karzmarz_True.mlp.a256x2x0.0e4x8.c256x2x0.0e4x8.npg_fisher_clip_0.5.vector.damping_0.1.lr_pi_0.05.lr_v_0.1"
N_EX_SWIM = LOGS / "rat.jointcritic.karzmarz_True.mlp.a256x2x0.0e4x8.c256x2x0.0e4x8.npg_fisher_clip_0.5.vector.damping_0.1.lr_pi_0.05.lr_v_0.2"
S_CU_KT_D003_LR05 = LOGS / "shared.rat.karzmarz_True.mlp.a256x2x0.0e4x8.clip_grad_0.5.vector.damping_0.03.lr_0.05.asub_256.csub_256"
S_CU_KT_D01_LR03 = LOGS / "shared.rat.karzmarz_True.mlp.a256x2x0.0e4x8.clip_grad_0.5.vector.damping_0.1.lr_0.03.asub_256.csub_256"
S_CU_KT_D01_LR05 = LOGS / "shared.rat.karzmarz_True.mlp.a256x2x0.0e4x8.clip_grad_0.5.vector.damping_0.1.lr_0.05.asub_256.csub_256"
N_CU_KT_D01_LRV001 = LOGS / "rat.jointcritic.karzmarz_True.mlp.a256x2x0.0e4x8.c256x2x0.0e4x8.npg_fisher_clip_0.5.vector.damping_0.1.lr_pi_0.05.lr_v_0.001.asub_256.csub_256"


def read_progress(path: Path) -> tuple[list[float], list[float]]:
    points = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                step = float(row.get("misc/total_timesteps", row.get("total_timesteps", "nan")))
                reward = float(row["eprewmean"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(step) and math.isfinite(reward):
                points[step] = reward
    steps = sorted(points)
    if not steps:
        raise RuntimeError(f"empty progress: {path}")
    return steps, [points[x] for x in steps]


def linked_progress(stdout: Path) -> Path:
    for line in stdout.read_text(errors="replace").splitlines():
        match = LOG_RE.match(line.strip())
        if match:
            path = REPO / match.group(1) / "progress.csv"
            if path.exists():
                return path
    raise RuntimeError(f"missing linked progress: {stdout}")


def formal_sources(root: Path, env: str, seeds: list[int]) -> list[Path]:
    return [linked_progress(root / env / f"seed{seed}.stdout") for seed in seeds]


def log_sources(root: Path, env: str, seeds: list[int]) -> tuple[list[Path], dict]:
    selected, duplicate_counts = [], {}
    for seed in seeds:
        matches = list(root.glob(f"{env}.*_{seed}/progress.csv"))
        choices = []
        for path in matches:
            try:
                steps, _ = read_progress(path)
            except RuntimeError:
                continue
            if steps[-1] < 9_500_000:
                continue
            choices.append((steps[-1], path.stat().st_mtime, path))
        if not choices:
            raise RuntimeError(f"missing log candidate {root}/{env}/seed{seed}")
        step, _, path = max(choices, key=lambda item: (item[0], item[1]))
        selected.append(path)
        duplicate_counts[str(seed)] = len(choices)
    return selected, duplicate_counts


def ppo_sources(mode: str, env: str) -> list[Path]:
    paths = sorted((PPO / "runs" / mode / env).glob("seed*_work/**/progress.csv"))
    if len(paths) != 5:
        raise RuntimeError(f"PPO {mode}/{env}: expected 5 paths, got {len(paths)}")
    return paths


def rolling10(values: list[float]) -> list[float]:
    return [sum(values[max(0, i-9):i+1]) / min(10, i+1) for i in range(len(values))]


def interpolate(steps: list[float], values: list[float]) -> list[float]:
    result, right = [], 0
    for target in GRID:
        if target <= steps[0]:
            result.append(values[0]); continue
        if target >= steps[-1]:
            result.append(values[-1]); continue
        while right + 1 < len(steps) and steps[right + 1] < target:
            right += 1
        x0, x1 = steps[right], steps[right+1]
        y0, y1 = values[right], values[right+1]
        result.append(y0 + (target-x0)/(x1-x0)*(y1-y0))
    return result


def aggregate(sources: list[Path]) -> tuple[dict, dict]:
    endpoints, finals, seed_curves = [], [], []
    for path in sources:
        steps, values = read_progress(path)
        smooth = rolling10(values)
        endpoints.append(smooth[-1])
        finals.append(steps[-1])
        seed_curves.append(interpolate(steps, smooth))
    n = len(endpoints)
    center = sum(endpoints) / n
    std = math.sqrt(sum((x-center)**2 for x in endpoints)/(n-1)) if n > 1 else 0.0
    curve_mean = [sum(c[i] for c in seed_curves)/n for i in range(len(GRID))]
    curve_min = [min(c[i] for c in seed_curves) for i in range(len(GRID))]
    curve_max = [max(c[i] for c in seed_curves) for i in range(len(GRID))]
    return ({
        "seed_count": n,
        "min_final_step": min(finals),
        "max_final_step": max(finals),
        "endpoint_last10_mean": center,
        "endpoint_last10_std": std,
        "endpoint_last10_stderr": std / math.sqrt(n),
        "endpoint_last10_min": min(endpoints),
        "endpoint_last10_max": max(endpoints),
        "seed_endpoint_last10": endpoints,
    }, {"grid_steps": GRID, "mean": curve_mean, "min": curve_min, "max": curve_max})


def status_for(root: Path, env: str, seeds: list[int]) -> dict:
    status = (root / env / "status").read_text().strip() if (root / env / "status").exists() else None
    rcs = [(root / env / f"seed{s}.rc").read_text().strip() if (root / env / f"seed{s}.rc").exists() else None for s in seeds]
    errors = []
    for path in (root / env).glob("seed*.stderr"):
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if ERROR_RE.search(line):
                errors.append(f"{path}:{number}:{line[:140]}")
    return {"status": status, "seed_rc": rcs, "error_hits": errors}


def exact_spec(arch: str, env: str) -> dict:
    if arch == "shared":
        root = EX_S_D01 if env == "halfcheetah" else EX_S_D003
        damping = .1 if root == EX_S_D01 else .03
        return dict(sources=formal_sources(root, env, list(range(5))), root=root, mode="direct_system", vf=1.0, damping=damping, kaczmarz=False, lr=.05, lr_v=None, seeds=list(range(5)))
    root = EX_N_D01 if env in {"halfcheetah", "hopper"} else EX_N_D003
    damping = .1 if root == EX_N_D01 else .03
    return dict(sources=formal_sources(root, env, list(range(5))), root=root, mode="independent", vf=None, damping=damping, kaczmarz=False, lr=.05, lr_v=.1, seeds=list(range(5)))


RHS_WINNERS = {
    "ant": ("trial_005_vf_2p0", 2.0),
    "halfcheetah": ("trial_006_vf_4p0", 4.0),
    "hopper": ("trial_006_vf_4p0", 4.0),
    "humanoidstandup": ("trial_004_vf_1p5", 1.5),
    "swimmer": ("trial_006_vf_4p0", 4.0),
    "walker2d": ("trial_002_vf_0p75", .75),
}


def curv_spec(arch: str, env: str) -> dict:
    if arch == "shared" and env in RHS_WINNERS:
        trial, vf = RHS_WINNERS[env]
        root = RHS_ROOT / trial
        seeds = [0, 1, 2]
        return dict(sources=formal_sources(root, env, seeds), root=root, mode="rhs_only", vf=vf, damping=.03, kaczmarz=False, lr=.05, lr_v=None, seeds=seeds, availability="partial_3seed_full10m")
    if arch == "shared":
        root = CU_S_D003
        return dict(sources=formal_sources(root, env, list(range(5))), root=root, mode="direct_system", vf=1.0, damping=.03, kaczmarz=False, lr=.05, lr_v=None, seeds=list(range(5)), availability="complete")
    root = CU_N_D01 if env in {"ant", "halfcheetah", "hopper"} else CU_N_D003
    damping = .1 if root == CU_N_D01 else .03
    return dict(sources=formal_sources(root, env, list(range(5))), root=root, mode="independent", vf=None, damping=damping, kaczmarz=False, lr=.05, lr_v=.1, seeds=list(range(5)), availability="complete")


def selection_record(spec: dict, method: str) -> dict:
    return {
        "method": method,
        "root": str(spec["root"]),
        "effect_mode": spec["mode"],
        "vf_coef": spec["vf"],
        "damping": spec["damping"],
        "kaczmarz": spec["kaczmarz"],
        "lr": spec["lr"],
        "critic_lr": spec["lr_v"],
        "seeds": spec["seeds"],
        "nominal_horizon": 10_000_000,
        "availability": spec.get("availability", "complete"),
        "duplicate_source_counts": spec.get("duplicate_counts"),
    }


def candidate_summary(label: str, sources: list[Path], extra: dict) -> dict:
    stats, _ = aggregate(sources)
    return {"label": label, **extra, **{k: stats[k] for k in ("seed_count", "min_final_step", "endpoint_last10_mean", "endpoint_last10_stderr")}}


def maybe_log_candidate(root: Path, env: str, label: str, extra: dict) -> dict | None:
    try:
        sources, duplicate_counts = log_sources(root, env, list(range(5)))
    except RuntimeError:
        return None
    return candidate_summary(label, sources, {"root": str(root), "duplicate_source_counts": duplicate_counts, **extra})


def main() -> None:
    table, curves, selections, validation = [], [], [], []
    for arch in ("shared", "no_shared"):
        ppo_mode = "shared" if arch == "shared" else "detach"
        for env in ENVS:
            methods = []
            ppo_spec = dict(sources=ppo_sources(ppo_mode, env), root=PPO, mode="shared_adam" if arch == "shared" else "independent_adam", vf=1.0 if arch == "shared" else None, damping=None, kaczmarz=False, lr=.0003, lr_v=.0003 if arch == "no_shared" else None, seeds=list(range(5)), availability="complete")
            methods.append(("ppo_fixed_lr", ppo_spec))
            methods.append(("exact_ef_ggn", exact_spec(arch, env)))
            methods.append(("ordinary_curv256_ggn256", curv_spec(arch, env)))
            for method, spec in methods:
                if spec["kaczmarz"]:
                    raise RuntimeError(f"Kaczmarz=true forbidden in main selection: {arch}/{env}/{method}")
                stats, curve = aggregate(spec["sources"])
                record = selection_record(spec, method)
                table.append({"architecture": arch, "environment": env, **record, **stats})
                curves.append({"architecture": arch, "environment": env, "method": method, "seed_count": stats["seed_count"], "metric": "per-seed rolling-last-10 eprewmean, linearly interpolated", **curve})
                selections.append({"architecture": arch, "environment": env, **record})
                validation.append({"architecture": arch, "environment": env, "method": method, "source_progress": [str(x) for x in spec["sources"]], "all_sources_exist": all(x.exists() for x in spec["sources"])})

    exact_candidates, curv_candidates = [], []
    for env in ENVS:
        for label, root, arch, damping in [
            ("shared_d003_kfalse_vf1_direct", EX_S_D003, "shared", .03),
            ("shared_d01_kfalse_vf1_direct", EX_S_D01, "shared", .1),
            ("no_shared_d003_kfalse", EX_N_D003, "no_shared", .03),
            ("no_shared_d01_kfalse", EX_N_D01, "no_shared", .1),
        ]:
            exact_candidates.append({"architecture": arch, "environment": env, **candidate_summary(label, formal_sources(root, env, list(range(5))), {"root": str(root), "effect_mode": "direct_system" if arch == "shared" else "independent", "vf_coef": 1.0 if arch == "shared" else None, "damping": damping, "kaczmarz": False, "lr": .05, "critic_lr": None if arch == "shared" else .1, "availability": "complete_5seed_10m"})})
        for label, root, arch, damping in [
            ("shared_d003_kfalse_vf1_direct", CU_S_D003, "shared", .03),
            ("shared_d01_kfalse_vf1_direct", CU_S_D01, "shared", .1),
            ("no_shared_d003_kfalse", CU_N_D003, "no_shared", .03),
            ("no_shared_d01_kfalse", CU_N_D01, "no_shared", .1),
        ]:
            curv_candidates.append({"architecture": arch, "environment": env, **candidate_summary(label, formal_sources(root, env, list(range(5))), {"root": str(root), "effect_mode": "direct_system" if arch == "shared" else "independent", "vf_coef": 1.0 if arch == "shared" else None, "damping": damping, "kaczmarz": False})})
        for trial in sorted(RHS_ROOT.glob("trial_*")):
            info = dict(line.split("=", 1) for line in (trial / "trial_info.txt").read_text().splitlines() if "=" in line)
            vf = float(info["vf_coef"])
            curv_candidates.append({"architecture": "shared", "environment": env, **candidate_summary(f"rhs_only_vf{vf:g}", formal_sources(trial, env, [0,1,2]), {"root": str(trial), "effect_mode": "rhs_only", "vf_coef": vf, "damping": .03, "kaczmarz": False, "availability": "partial_3seed_full10m"})})
        for root, label, damping, lr in [
            (S_CU_KT_D003_LR05, "shared_d003_ktrue_lr005_vf1_direct", .03, .05),
            (S_CU_KT_D01_LR03, "shared_d01_ktrue_lr003_vf1_direct", .1, .03),
            (S_CU_KT_D01_LR05, "shared_d01_ktrue_lr005_vf1_direct", .1, .05),
        ]:
            candidate = maybe_log_candidate(root, env, label, {"effect_mode": "direct_system", "vf_coef": 1.0, "damping": damping, "kaczmarz": True, "lr": lr})
            if candidate:
                curv_candidates.append({"architecture": "shared", "environment": env, **candidate})
        candidate = maybe_log_candidate(N_CU_KT_D01_LRV001, env, "no_shared_d01_ktrue_lrpi005_lrv0001", {"effect_mode": "independent", "vf_coef": None, "damping": .1, "kaczmarz": True, "lr": .05, "critic_lr": .001})
        if candidate:
            curv_candidates.append({"architecture": "no_shared", "environment": env, **candidate})

    exact_vf_screen_results = []
    for label, vf in [("vf0p1",.1),("vf0p25",.25),("vf0p5",.5),("vf0p75",.75),("vf1p5",1.5),("vf2p0",2.0),("vf4p0",4.0),("vf8p0",8.0)]:
        sources = [linked_progress(EXACT_VF_SCREEN / "stdout" / f"{label}_seed{seed}.out") for seed in [0,1,2]]
        exact_vf_screen_results.append(candidate_summary(f"direct_system_vf{vf:g}", sources, {"vf_coef": vf, "effect_mode": "direct_system", "damping": .1, "kaczmarz": False, "lr": .05, "availability": "partial_3seed_6m_screen"}))

    document = {
        "schema_version": 2,
        "generated_utc": os.environ.get("AUDIT_GENERATED_UTC", "2026-07-19"),
        "selection_rule": "Hard constraint Kaczmarz=false. Within that constraint, select the highest endpoint-last10 formal candidate separately per environment. Full-10M 3-seed sweeps may win but remain explicitly partial; screens below 10M never replace a 10M line.",
        "curve_rule": "0..10M every 0.5M; per-seed rolling-last-10 eprewmean; linear interpolation; endpoint hold",
        "vf_semantics": {
            "direct_system": "train_shared_jointkernel.py: joint_H=[H_pi;sqrt(vf)J_v], joint_rhs=[A;sqrt(vf)(R-V)]; vf changes both Gram and RHS",
            "rhs_only": "train_shared_jointkernel_vfrhs.py: joint_H=[H_pi;J_v], joint_rhs=[A;vf(R-V)]; vf changes only critic RHS",
            "independent": "no-shared actor/critic solves; shared vf coupling is not applicable",
        },
        "fixed_ppo_provenance": {"archive": str(PPO), "source_jobs": [17595177,17595178], "lr": .0003, "adaptive_kl": False, "epochs_x_minibatches": "4x8", "horizon": 10_000_000, "seeds": [0,1,2,3,4]},
        "excluded_families": ["Energy-free255+1", "full-gradient-256", "free-rho", "grouped-rho", "hierarchical"],
        "hard_constraints": {"kaczmarz": False, "main_table_ktrue_count": 0},
        "partial_or_screen_only": [{
            "family": "shared exact EF+GGN direct-system VF screen",
            "environment": "hopper",
            "root": str(EXACT_VF_SCREEN),
            "trainer": "train_shared_jointkernel_vfsweep.py",
            "vf_candidates": [.1,.25,.5,.75,1.5,2,4,8],
            "seeds": [0,1,2],
            "horizon": 6_000_000,
            "disposition": "screen_only_not_selected_as_10m",
            "candidate_endpoints": exact_vf_screen_results,
        }, {
            "family": "archived shared curv256 direct-system VF t26/t27",
            "vf_candidates": [.5,2.0],
            "seeds": [0,1,2],
            "horizon": 4_000_000,
            "disposition": "screen_only_not_selected; archive summaries report no_eprewmean",
        }],
        "selection": selections,
        "candidate_audit_exact_kfalse": exact_candidates,
        "candidate_audit_curv256": curv_candidates,
        "candidate_audit_notes": "K=true entries are included only when all five seeds reached at least 9.5M. Missing K=true environment/config combinations are incomplete and were not eligible.",
        "table": table,
        "curves": curves,
        "validation": validation,
    }
    print(json.dumps(document, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
