#!/usr/bin/env python3
"""Audit completed exact-EF or ordinary-curv256 + GGN MuJoCo candidates."""

from __future__ import annotations

import csv
import glob
import math
import os
import re
from collections import defaultdict


REPO = "/scratch/h99859yz/rat_default_shared_csf3/rat_repos/ICML2026-RAT-original"
LOG_ROOT = os.path.join(REPO, "logs")
FAMILY = os.environ.get("RAT_AUDIT_FAMILY", "exact")
ENVS = [
    "ant",
    "halfcheetah",
    "hopper",
    "humanoid",
    "humanoidstandup",
    "swimmer",
    "walker2d",
]


def read_curve(path: str) -> tuple[list[float], list[float]]:
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
    return steps, rewards


def classify(name: str) -> str | None:
    is_curv256 = ".asub_256.csub_256" in name
    if FAMILY == "exact" and (".asub_" in name or ".csub_" in name):
        return None
    if FAMILY == "curv256" and not is_curv256:
        return None
    if name.startswith("shared.rat.karzmarz_"):
        return "shared"
    if name.startswith("rat.jointcritic.karzmarz_"):
        return "no_shared"
    return None


rows = []
for method_dir in glob.glob(os.path.join(LOG_ROOT, "*")):
    method = os.path.basename(method_dir)
    mode = classify(method)
    if mode is None or "e4x8" not in method:
        continue
    candidates = defaultdict(list)
    for path in glob.glob(os.path.join(method_dir, "*", "progress.csv")):
        run = os.path.basename(os.path.dirname(path)).lower()
        env = next((item for item in sorted(ENVS, key=len, reverse=True) if run.startswith(item + ".")), None)
        seed_match = re.search(r"_([0-9]+)$", run)
        if env is None or seed_match is None:
            continue
        seed = int(seed_match.group(1))
        if seed not in range(5):
            continue
        steps, rewards = read_curve(path)
        if steps:
            candidates[(env, seed)].append((steps[-1], os.path.getmtime(path), rewards))
    grouped = defaultdict(list)
    final_steps = defaultdict(list)
    for (env, seed), choices in candidates.items():
        step, _, rewards = max(choices, key=lambda item: (item[0], item[1]))
        if step < 9.5e6:
            continue
        grouped[env].append(sum(rewards[-10:]) / min(10, len(rewards)))
        final_steps[env].append(step)
    for env, values in grouped.items():
        if len(values) != 5:
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        stderr = math.sqrt(variance / len(values))
        rows.append((mode, env, mean, stderr, min(final_steps[env]), method))

print("mode\tenv\tmean_last10\tstderr\tmin_final_steps\tmethod")
for row in sorted(rows, key=lambda item: (item[0], item[1], -item[2])):
    print("\t".join([row[0], row[1], f"{row[2]:.6f}", f"{row[3]:.6f}", f"{row[4]:.0f}", row[5]]))
