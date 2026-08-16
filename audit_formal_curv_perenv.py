#!/usr/bin/env python3
"""Read-only endpoint audit for formal curv256 roots and RHS-only trials."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

REPO = Path("/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original")
RUNS = REPO / "perf_runs"
ENVS = ["ant", "halfcheetah", "hopper", "humanoid", "humanoidstandup", "swimmer", "walker2d"]
LOG_RE = re.compile(r"^Logging to\s+(\S+)\s*$")
ROOTS = {
    "shared_d003_kfalse_vf1_direct": RUNS / "csf3_shared_curv256_ggn_normnone_damp003_kfalse_all7_5seed_10m_gpuL2_17505797",
    "shared_d01_kfalse_vf1_direct": RUNS / "csf3_shared_curv256_ggn_normnone_damp01_kfalse_all7_5seed_10m_gpuL2_17505023",
    "no_shared_d003_kfalse_vf1_direct": RUNS / "csf3_detjc_curv256_ggn_normnone_damp003_kfalse_all7_5seed_10m_gpuL2_17491618",
    "no_shared_d01_kfalse_vf1_direct": RUNS / "csf3_detjc_curv256_ggn_normnone_damp01_kfalse_all7_5seed_10m_gpuL2_17510356",
}
RHS = RUNS / "csf3_shared_vfrhs_curv256_optuna_d003_kfalse_10m_3seed_20260713" / "trials"


def progress_from_stdout(stdout: Path) -> Path:
    for line in stdout.read_text(errors="replace").splitlines():
        match = LOG_RE.match(line.strip())
        if match:
            path = REPO / match.group(1) / "progress.csv"
            if path.exists():
                return path
    raise RuntimeError(f"no linked progress: {stdout}")


def endpoint(path: Path) -> tuple[float, float]:
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
    values = [points[step] for step in steps]
    return steps[-1], sum(values[-10:]) / min(10, len(values))


def report(label: str, root: Path, seeds: list[int]) -> None:
    for env in ENVS:
        values, final_steps = [], []
        for seed in seeds:
            stdout = root / env / f"seed{seed}.stdout"
            step, value = endpoint(progress_from_stdout(stdout))
            final_steps.append(step)
            values.append(value)
        center = sum(values) / len(values)
        stderr = (sum((x-center)**2 for x in values)/(len(values)-1)/len(values))**0.5 if len(values)>1 else 0
        print(f"{label}\t{env}\t{len(values)}\t{min(final_steps):.0f}\t{center:.6f}\t{stderr:.6f}\t{root}")


print("candidate\tenv\tn\tmin_final_step\tmean_last10\tstderr\troot")
for label, root in ROOTS.items():
    report(label, root, list(range(5)))
for trial in sorted(RHS.glob("trial_*")):
    if (trial / "status").read_text().strip() != "FINISHED":
        continue
    report(f"shared_{trial.name}_rhs_only", trial, [0, 1, 2])
